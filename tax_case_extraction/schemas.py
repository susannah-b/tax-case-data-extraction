from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict, Any
from datetime import date


class LegislationReference(BaseModel):
    """Reference to a piece of legislation with relationship context."""
    name: str = Field(..., description="Name of the statute, treaty, or regulation (e.g., 'Finance Act 2002', 'TCGA 1992 s162').")
    relationship: str = Field(..., description="How the case relates to this legislation. Options: 'applied', 'interpreted', 'cited', 'overruled', 'referred_to'")
    context: Optional[str] = Field(None, description="Brief context of how/why this legislation was used in the case.")


class TaxCaseMetadata(BaseModel):
    case_name: str = Field(..., description="Full name of the case, including parties (e.g., 'Carulla Font v HMRC').")
    neutral_citation: str = Field(..., description="Neutral citation reference (e.g., '[2025] EWHC 3057 (Admin)').")
    case_number: str = Field(..., description="Official case number as stated in the judgment.")
    court_name: str = Field(..., description="Name of the court or tribunal (e.g., 'High Court of Justice, King's Bench Division').")
    judgment_date: Optional[date] = Field(None, description="Date the judgment was handed down.")
    hearing_dates: List[date] = Field(..., description="List of hearing dates mentioned in the case.")
    judges: List[str] = Field(..., description="Names of the judge(s) or tribunal members who decided the case.")
    parties: List[str] = Field(..., description="Names of claimant/appellant and defendant/respondent. MUST Identify which party is claimant/appellant and defendant/respondent.")
    representation: List[str] = Field(..., description="Counsel and firms representing each party.")
    citation_links: Optional[List[str]] = Field(None, description="Links to cited legislation or treaties if available.")

class TaxCaseFacts(BaseModel):
    detailed_facts: str = Field(..., description="Comprehensive summary of the facts, MUST include a chronology and relevant events.")
    key_dates: Optional[List[str]] = Field(None, description="Important dates related to the facts (e.g., transaction dates, correspondence). MUST include what happened on those dates.")

class TaxCaseLegislation(BaseModel):
    legislation_list: List[LegislationReference] = Field(..., description="List of statutes, treaties, or regulations referenced or applied in the case, with relationship types.")

class TaxCaseOverview(BaseModel):
    overview: str = Field(..., description="Concise summary of the case background, issues, and context.")

class TaxCaseJudgesComments(BaseModel):
    dicta: str = Field(..., description="Judicial observations, reasoning, and comments (dicta) relevant to interpretation and principles.")
    reasoning: str = Field(..., description="Detailed explanation of how the judges reached their conclusions, MUST include analysis of evidence, logical steps, and evaluative process.")

class TaxCaseDecision(BaseModel):
    conclusion: str = Field(..., description="Final decision or outcome of the case (e.g., permission refused, appeal dismissed).")
    reasoning_summary: Optional[str] = Field(None, description="Brief summary of the reasoning behind the decision.")

class TaxCaseExtraction(BaseModel):
    metadata: TaxCaseMetadata
    facts: TaxCaseFacts
    legislation: TaxCaseLegislation
    overview: TaxCaseOverview
    judges_comments: TaxCaseJudgesComments
    decision: TaxCaseDecision
