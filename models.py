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

class SubTopic(BaseModel):
    name: str = Field(description="The name of the specific subtopic or angle")
    importance: str = Field(description="Why this subtopic is critical to understanding the main topic")
    target_search_query: str = Field(description="An optimized, highly-specific arXiv search query for this subtopic")

class TopicPlan(BaseModel):
    main_topic: str = Field(description="The primary research topic")
    conceptual_summary: str = Field(description="A brief paragraph summarizing what makes this field complex")
    subtopics: List[SubTopic] = Field(description="A list of exactly 4 critical subtopics to map the field completely")