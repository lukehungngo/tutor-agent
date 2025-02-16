from typing import Annotated, Optional, List, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]


class BaseAnswer(BaseModel):
    summary: str

    def model_dump(self, **kwargs) -> dict:
        return {
            "summary": self.summary,
        }


class ResearchNode(BaseModel):
    title: str = Field(description="The title of the resource")
    content: Optional[str] = Field(None, description="A summary of the resource's content")
    url: Optional[str] = Field(None, description="The URL of the resource")
    category: Optional[str] = Field(None, description="The category of the resource")
    author_or_source: Optional[str] = Field(None, description="The author or source of the resource")
    difficulty_level: Optional[str] = Field(None, description="The difficulty level of the course or book")
    relevance_score: Optional[float] = Field(None, description="Relevance score from 0-1")
    confidence_score: Optional[float] = Field(None, description="Confidence score from 0-1")
    source_type: Optional[str] = Field(None, description="Type of source (e.g., Academic Paper)")
    last_updated: Optional[str] = Field(None, description="Last updated date")

class ResearchResult(BaseAnswer):
    research_nodes: Optional[List[ResearchNode]]
    
    def model_dump(self, **kwargs) -> dict:
        return {
            "summary": self.summary,
            "research_nodes": [node.model_dump() for node in self.research_nodes] if self.research_nodes else None
        }
    
class ResearchState(TypedDict, total=False):
    messages: Annotated[List, add_messages]
    structured_output: Optional[ResearchResult]
