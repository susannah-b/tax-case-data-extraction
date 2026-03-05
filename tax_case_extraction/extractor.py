"""
Tax Case Extraction Pipeline
Extracts structured information from tax case judgment PDFs using LLM-based extraction.
"""
import os
import re
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF for PDF reading
import instructor
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .schemas import TaxCaseExtraction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# -----------------------------
# Constants
# -----------------------------
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_OVERLAP = 250
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))
MAX_WORKERS = 3  # Parallel processing for chunks (be mindful of API rate limits)
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1")  # Configurable model

# -----------------------------
# Custom JSON Encoder for date objects
# -----------------------------
class DateEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles date and datetime objects."""
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


# -----------------------------
# Caching & Utilities
# -----------------------------
def get_file_hash(file_path: str) -> str:
    """
    Generate MD5 hash of a file for caching purposes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_cache_path(pdf_path: str, stage: str = "text") -> Path:
    """
    Get cache file path for a PDF at a specific processing stage.
    
    Args:
        pdf_path: Path to the PDF file
        stage: Processing stage ("text" or "extraction")
        
    Returns:
        Path to cache file
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    file_hash = get_file_hash(pdf_path)
    return CACHE_DIR / f"{Path(pdf_path).stem}_{file_hash}_{stage}.json"


def load_from_cache(cache_path: Path) -> Optional[Any]:
    """
    Load data from cache if it exists and is valid.
    
    Args:
        cache_path: Path to cache file
        
    Returns:
        Cached data or None if not found/invalid
    """
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded data from cache: {cache_path.name}")
        return data
    except Exception as e:
        logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None


def save_to_cache(cache_path: Path, data: Any) -> None:
    """
    Save data to cache.
    
    Args:
        cache_path: Path to cache file
        data: Data to cache
    """
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=DateEncoder, ensure_ascii=False)
        logger.debug(f"Saved data to cache: {cache_path.name}")
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_path}: {e}")

# -----------------------------
# Utility Functions
# -----------------------------
def extract_text_from_pdf(pdf_path: str, use_cache: bool = True) -> str:
    """
    Extract text from a PDF using PyMuPDF with caching support.
    
    Args:
        pdf_path: Path to the PDF file
        use_cache: Whether to use cached text if available
        
    Returns:
        Extracted text content
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        RuntimeError: If PDF extraction fails
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Check cache first
    if use_cache:
        cache_path = get_cache_path(pdf_path, "text")
        cached_text = load_from_cache(cache_path)
        if cached_text and isinstance(cached_text, dict):
            return cached_text.get("text", "")
    
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        page_count = len(doc)
        
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text("text")
            text_parts.append(page_text)
            if page_num % 10 == 0:  # Log every 10 pages
                logger.debug(f"Extracted {page_num}/{page_count} pages from {os.path.basename(pdf_path)}")
        doc.close()
        
        full_text = "\n".join(text_parts)
        logger.info(f"Successfully extracted {len(full_text):,} characters from {pdf_path} ({page_count} pages)")
        
        # Save to cache
        if use_cache:
            cache_path = get_cache_path(pdf_path, "text")
            save_to_cache(cache_path, {"text": full_text, "page_count": page_count})
        
        return full_text
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        raise RuntimeError(f"PDF extraction failed: {e}") from e

def split_into_sections(
    text: str, 
    chunk_size: int = DEFAULT_CHUNK_SIZE, 
    overlap: int = DEFAULT_OVERLAP
) -> List[str]:
    """
    Split text into chunks with overlap, preserving original formatting.
    
    Args:
        text: Text to split into chunks
        chunk_size: Number of words per chunk (default: 1500)
        overlap: Number of overlapping words between chunks (default: 250)
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If chunk_size <= overlap or invalid parameters
    """
    if chunk_size <= overlap:
        raise ValueError(f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})")
    
    if not text or not text.strip():
        logger.warning("Empty or whitespace-only text provided")
        return []

    # Find spans of non-whitespace tokens to preserve original character offsets
    token_spans = [m.span() for m in re.finditer(r'\S+', text)]
    total_tokens = len(token_spans)
    
    if total_tokens == 0:
        return []
    
    if total_tokens <= chunk_size:
        logger.info(f"Text has {total_tokens} tokens, returning as single chunk")
        return [text.strip()]

    step = chunk_size - overlap
    chunks: List[str] = []
    start_word = 0

    while start_word < total_tokens:
        end_word = min(start_word + chunk_size - 1, total_tokens - 1)
        start_char = token_spans[start_word][0]
        end_char = token_spans[end_word][1]
        chunk = text[start_char:end_char].strip()
        
        if chunk:
            chunks.append(chunk)
            logger.debug(f"Created chunk {len(chunks)} with {len(chunk)} characters")
        
        if end_word == total_tokens - 1:
            break
        start_word += step

    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def merge_extraction_data(base_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two extraction dictionaries intelligently.
    
    Args:
        base_data: Existing data dictionary
        new_data: New data to merge in
        
    Returns:
        Merged dictionary
    """
    for field, value in new_data.items():
        if field not in base_data or not base_data[field]:
            base_data[field] = value
        elif isinstance(value, list) and isinstance(base_data[field], list):
            # Extend lists, avoiding duplicates for simple types
            for item in value:
                if item not in base_data[field]:
                    base_data[field].append(item)
        elif isinstance(value, str) and isinstance(base_data[field], str):
            # Concatenate strings if they're different
            if value.strip() and value.strip() not in base_data[field]:
                base_data[field] += "\n" + value
        elif isinstance(value, dict) and isinstance(base_data[field], dict):
            # Recursively merge nested dictionaries
            base_data[field] = merge_extraction_data(base_data[field], value)
    
    return base_data


def deduplicate_lists(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively deduplicate all lists in the data structure while preserving order.
    
    Args:
        data: Dictionary containing potentially duplicate list items
        
    Returns:
        Dictionary with deduplicated lists
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            result[key] = deduplicate_lists(value)
        return result
    elif isinstance(data, list):
        # Preserve order while removing duplicates
        seen = set()
        deduped = []
        for item in data:
            # For unhashable types (dicts, lists), convert to string for comparison
            if isinstance(item, (dict, list)):
                item_str = json.dumps(item, sort_keys=True, cls=DateEncoder)
                if item_str not in seen:
                    seen.add(item_str)
                    deduped.append(deduplicate_lists(item))
            else:
                # For hashable types (strings, numbers, dates, etc.)
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
        
        duplicates_removed = len(data) - len(deduped)
        if duplicates_removed > 0:
            logger.debug(f"Removed {duplicates_removed} duplicate(s) from list")
        
        return deduped
    else:
        return data


def validate_extraction_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate extracted data for completeness and quality.
    
    Args:
        data: Extracted data dictionary
        
    Returns:
        Tuple of (is_valid, list of warnings/issues)
    """
    issues = []
    
    # Check for required top-level fields
    required_fields = ["metadata", "facts", "legislation", "overview", "judges_comments", "decision"]
    for field in required_fields:
        if field not in data or not data[field]:
            issues.append(f"Missing or empty required field: {field}")
    
    # Check metadata completeness
    if "metadata" in data and isinstance(data["metadata"], dict):
        metadata_fields = ["case_name", "neutral_citation", "court_name", "judgment_date"]
        for field in metadata_fields:
            if not data["metadata"].get(field):
                issues.append(f"Missing metadata field: {field}")
    
    # Warn if decision is too short
    if "decision" in data and isinstance(data["decision"], dict):
        conclusion = data["decision"].get("conclusion", "")
        if isinstance(conclusion, str) and len(conclusion) < 20:
            issues.append("Decision conclusion seems too short")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def save_extraction_results(
    pdf_path: str, 
    data: Dict[str, Any], 
    output_dir: Path = OUTPUT_DIR,
    output_filename: Optional[str] = None,
    add_metadata: bool = True,
    deduplicate: bool = True
) -> Path:
    """
    Save extraction results to a JSON file with optional metadata and deduplication.
    
    Args:
        pdf_path: Original PDF file path
        data: Extracted data dictionary
        output_dir: Directory to save output files
        output_filename: Optional custom output filename (without extension)
        add_metadata: Whether to add processing metadata
        deduplicate: Whether to deduplicate lists before saving
        
    Returns:
        Path to the saved JSON file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Deduplicate lists if requested
    if deduplicate:
        logger.info("Deduplicating lists in extraction data...")
        data = deduplicate_lists(data)
    
    # Add processing metadata
    if add_metadata:
        data["_processing_metadata"] = {
            "source_file": str(pdf_path),
            "processed_at": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "version": "1.0"
        }
    
    if output_filename:
        json_filename = f"{output_filename}.json"
    else:
        json_filename = Path(pdf_path).stem + "_extraction-"+MODEL_NAME+".json"
    output_path = output_dir / json_filename
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, cls=DateEncoder, ensure_ascii=False)
        
        file_size = output_path.stat().st_size
        logger.info(f"Saved extraction results to {output_path} ({file_size:,} bytes)")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")
        raise


@lru_cache(maxsize=1)
def get_instructor_client(api_key: str = None):
    """
    Initialize and return an instructor client (cached singleton).
    
    Args:
        api_key: OpenAI API key (uses env variable if not provided)
        
    Returns:
        Configured instructor client
        
    Raises:
        ValueError: If API key is not available
    """
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    
    base_url = os.getenv('OPENAI_BASE_URL')
    
    try:
        client_kwargs = {
            "api_key": api_key,
        }
        
        if base_url:
            client_kwargs["base_url"] = base_url
            logger.info(f"Using custom base URL: {base_url}")
        
        client = instructor.from_provider(
            f"openai/{MODEL_NAME}",
            **client_kwargs
        )
        logger.info("Successfully initialized instructor client")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize instructor client: {e}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True
)
def extract_single_chunk(client, chunk: str, chunk_idx: int, total_chunks: int, pdf_name: str) -> Dict[str, Any]:
    """
    Extract structured data from a single text chunk with retry logic.
    
    Args:
        client: Instructor client
        chunk: Text chunk to process
        chunk_idx: Current chunk index (1-based)
        total_chunks: Total number of chunks
        pdf_name: Name of the PDF being processed
        
    Returns:
        Extracted data dictionary
        
    Raises:
        Exception: If extraction fails after retries
    """
    logger.info(f"Processing chunk {chunk_idx}/{total_chunks} from {pdf_name}")
    
    try:
        # Only set temperature if model is not o1/o3 (reasoning models don't support it)
        model_params = {
            "model": MODEL_NAME,
            "messages": [
            {
                "role": "system", 
                "content": "You are an expert legal assistant extracting structured data from tax case judgments. "
                       "Extract all relevant information accurately and comprehensively. "
                       "If information is not present in this chunk, You MUST leave those fields empty."
                       "DO NOT make up any information."
                       "Only provide information that is explicitly stated in the text."
                       "If you are unsure about any information, You MUST leave it blank."
                       "You MUST leave it blank, if you cannot find it in the text."
            },
            {
                "role": "user", 
                "content": f"Extract the following structured information from this section of a tax case judgment below"
                       f"(chunk {chunk_idx}/{total_chunks}):\n\n{chunk}"
            }
            ],
            "response_model": TaxCaseExtraction,
            "max_retries": 3,
        }
        
        # Only add temperature for non-reasoning models
        if not any(x in MODEL_NAME.lower() for x in ['o1', 'o3','gpt-5']):
            model_params["temperature"] = 0.3
        
        extracted_chunk = client.chat.completions.create(**model_params)
        
        logger.debug(f"Successfully processed chunk {chunk_idx}/{total_chunks}")
        return extracted_chunk.model_dump()
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_idx}/{total_chunks} from {pdf_name}: {e}")
        raise


def extract_from_chunks(
    client, 
    chunks: List[str], 
    pdf_name: str,
    parallel: bool = False
) -> Dict[str, Any]:
    """
    Extract structured data from text chunks using LLM.
    
    Args:
        client: Instructor client
        chunks: List of text chunks to process
        pdf_name: Name of the PDF being processed (for logging)
        parallel: Whether to process chunks in parallel (use with caution for rate limits)
        
    Returns:
        Merged extraction data dictionary
    """
    merged_data = {}
    total_chunks = len(chunks)
    
    if parallel and total_chunks > 1:
        logger.info(f"Processing {total_chunks} chunks in parallel (max {MAX_WORKERS} workers)")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all chunks for processing
            future_to_idx = {
                executor.submit(extract_single_chunk, client, chunk, idx, total_chunks, pdf_name): idx
                for idx, chunk in enumerate(chunks, 1)
            }
            
            # Process completed chunks as they finish
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    chunk_dict = future.result()
                    merged_data = merge_extraction_data(merged_data, chunk_dict)
                except Exception as e:
                    logger.error(f"Chunk {idx} failed: {e}")
                    continue
    else:
        # Sequential processing
        for idx, chunk in enumerate(chunks, 1):
            try:
                chunk_dict = extract_single_chunk(client, chunk, idx, total_chunks, pdf_name)
                merged_data = merge_extraction_data(merged_data, chunk_dict)
            except Exception as e:
                logger.error(f"Error processing chunk {idx}/{total_chunks} from {pdf_name}: {e}")
                continue
    
    return merged_data


def process_pdf_file(
    pdf_path: str, 
    client=None,
    use_cache: bool = True,
    parallel_chunks: bool = False
) -> Dict[str, Any]:
    """
    Process a single PDF file: extract text, chunk, and extract structured data.
    
    Args:
        pdf_path: Path to the PDF file
        client: Instructor client (will be initialized if not provided)
        use_cache: Whether to use cached results if available
        parallel_chunks: Whether to process chunks in parallel
        
    Returns:
        Extracted structured data dictionary
    """
    logger.info(f"Starting processing of {pdf_path}")
    
    # Initialize client if not provided
    if client is None:
        client = get_instructor_client()
    
    # Check if we have cached extraction results
    if use_cache:
        cache_path = get_cache_path(pdf_path, "extraction")
        cached_data = load_from_cache(cache_path)
        if cached_data:
            logger.info(f"Using cached extraction for {pdf_path}")
            return cached_data
    
    # Extract text from PDF (with its own caching)
    text = extract_text_from_pdf(pdf_path, use_cache=use_cache)
    
    # Split into chunks
    chunks = split_into_sections(text)
    logger.info(f"Split {pdf_path} into {len(chunks)} chunks")
    
    # Extract from chunks
    extracted_data = extract_from_chunks(
        client, 
        chunks, 
        os.path.basename(pdf_path),
        parallel=parallel_chunks
    )
    
    # Validate extraction
    is_valid, issues = validate_extraction_data(extracted_data)
    if not is_valid:
        logger.warning(f"Extraction validation issues for {pdf_path}:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    # Save to cache
    if use_cache:
        cache_path = get_cache_path(pdf_path, "extraction")
        save_to_cache(cache_path, extracted_data)
    
    logger.info(f"Completed processing of {pdf_path}")
    return extracted_data


def discover_pdf_files(directory: Path = DOCS_DIR, pattern: str = "*.pdf") -> List[Path]:
    """
    Discover PDF files in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        
    Returns:
        List of PDF file paths
    """
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []
    
    pdf_files = list(directory.glob(pattern))
    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
    return pdf_files


def extract(
    pdf_path: Optional[str] = None,
    pdf_files: Optional[List[str]] = None,
    use_cache: bool = True,
    parallel_chunks: bool = False,
    output_dir: Path = OUTPUT_DIR
):
    """
    Main extraction function - extracts structured data from tax case PDFs.
    
    Args:
        pdf_path: Single PDF file path to process (most common usage)
        pdf_files: List of PDF file paths to process (alternative to pdf_path)
        use_cache: Whether to use cached results
        parallel_chunks: Whether to process chunks in parallel
        output_dir: Directory to save output files
        
    Note: If both pdf_path and pdf_files are None, auto-discovers PDFs from docs/ directory
    """
    start_time = datetime.now()
    logger.info(f"=== Starting Tax Case Extraction Pipeline ===")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Cache enabled: {use_cache}")
    logger.info(f"Parallel processing: {parallel_chunks}")
    
    try:
        # Initialize client
        client = get_instructor_client()
        
        # Define or discover PDF files to process
        if pdf_path is not None:
            # Single file path provided
            pdf_paths = [Path(pdf_path)]
        elif pdf_files is not None:
            # List of files provided
            pdf_paths = [Path(p) for p in pdf_files]
        else:
            # Auto-discover from docs directory
            pdf_paths = discover_pdf_files(DOCS_DIR)
        
        # Verify files exist
        valid_pdf_files = []
        for pdf_path in pdf_paths:
            if not pdf_path.exists():
                logger.warning(f"PDF file not found: {pdf_path}")
            else:
                valid_pdf_files.append(pdf_path)
        
        if not valid_pdf_files:
            logger.error("No valid PDF files found to process")
            return {}
        
        logger.info(f"Processing {len(valid_pdf_files)} PDF files")
        
        # Process each PDF
        results = {}
        successful = 0
        failed = 0
        
        for pdf_path in valid_pdf_files:
            try:
                extracted_data = process_pdf_file(
                    str(pdf_path), 
                    client,
                    use_cache=use_cache,
                    parallel_chunks=parallel_chunks
                )
                results[str(pdf_path)] = extracted_data
                
                # Save results immediately after each file
                save_extraction_results(str(pdf_path), extracted_data, output_dir=output_dir)
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}", exc_info=True)
                failed += 1
                continue
        
        # Summary
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"=== Extraction Complete ===")
        logger.info(f"Successful: {successful}/{len(valid_pdf_files)}")
        logger.info(f"Failed: {failed}/{len(valid_pdf_files)}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Output directory: {output_dir.absolute()}")
        
        return results
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}", exc_info=True)
        raise
