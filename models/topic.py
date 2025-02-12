from typing import List, Optional
from pydantic import BaseModel, Field

class Subtopic(BaseModel):
    """Schema for an individual subtopic"""
    title: str = Field(...)  # example="What are the fundamental principles of photosynthesis?"
    description: Optional[str] = Field(None)  # example="This subtopic covers the basic principles of photosynthesis, including the role of chlorophyll and the process of converting light energy into chemical energy."
    category: Optional[str] = Field(None)  # Beginner, Intermediate, Advanced
    relevance_score: Optional[float] = Field(None)  # AI-generated importance score (0-1)
    sources: Optional[List[str]] = Field(None)  # Research sources
    recommended_resources: Optional[List[str]] = Field(None)  # Links to further learning

class Subtopics(BaseModel):
    """Schema for structured subtopics"""
    task: str = Field(...) #  example="How does photosynthesis work?"
    subtopics: List[Subtopic]  # List of subtopics with metadata
