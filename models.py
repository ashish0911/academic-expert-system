from pydantic import BaseModel, Field
from typing import List

class CitedSource(BaseModel):
    citation_number: int = Field(description="The number [1], [2], etc.")
    title: str = Field(description="The title of the paper")
    url: str = Field(description="The PDF URL")
    arxiv_id: str = Field(description="The arXiv ID")

class ExpertAnswer(BaseModel):
    answer_text: str = Field(description="The answer with simple numeric citations like [1]")
    sources_used: List[CitedSource] = Field(description="List of sources used")

class SearchRequest(BaseModel):
    queries: List[str] = Field(description="3 distinct search queries for arXiv")