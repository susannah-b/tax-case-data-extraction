"""
Tax Case Extraction Package
Extracts structured information from tax case judgment PDFs using LLM-based extraction.
"""

from .extractor import (
    extract,
    process_pdf_file,
    extract_text_from_pdf,
    get_instructor_client,
    DateEncoder,
    TaxCaseExtraction,
)

__version__ = "1.0.0"

__all__ = [
    "extract",
    "process_pdf_file",
    "extract_text_from_pdf",
    "get_instructor_client",
    "DateEncoder",
    "TaxCaseExtraction",
]
