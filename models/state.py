from typing import Annotated, Optional, List, TypedDict, Union
from pydantic import BaseModel, Field
from langgraph.graph import add_messages


class ResearchNode(BaseModel):
    title: str = Field(description="The title of the resource")
    content: Optional[str] = Field(
        None, description="A summary of the resource's content"
    )
    url: Optional[str] = Field(None, description="The URL of the resource")
    category: Optional[str] = Field(None, description="The category of the resource")
    author_or_source: Optional[str] = Field(
        None, description="The author or source of the resource"
    )
    difficulty_level: Optional[str] = Field(
        None, description="The difficulty level of the course or book"
    )
    relevance_score: Optional[float] = Field(
        None, description="Relevance score from 0-1"
    )
    confidence_score: Optional[float] = Field(
        None, description="Confidence score from 0-1"
    )
    source_type: Optional[str] = Field(
        None, description="Type of source (e.g., Academic Paper)"
    )
    last_updated: Optional[str] = Field(None, description="Last updated date")


class ResearchResult(BaseModel):
    summary: str
    research_nodes: Optional[List[ResearchNode]]

    def model_dump(self, **kwargs) -> dict:
        return {
            "summary": self.summary,
            "research_nodes": (
                [node.model_dump() for node in self.research_nodes]
                if self.research_nodes
                else None
            ),
        }

class ReasoningResult(BaseModel):
    question: str = Field(description="The original question or task being reasoned about")
    reasoning_process: str = Field(description="Detailed reasoning steps and analysis")
    confidence_level: int = Field(description="Confidence level from 1-10")
    used_context: bool = Field(description="Whether external context was used in reasoning")

    def model_dump(self, **kwargs) -> dict:
        return {
            "question": self.question,
            "reasoning_process": self.reasoning_process,
            "confidence_level": self.confidence_level,
            "used_context": self.used_context
        }

class State(TypedDict, total=False):
    messages: Annotated[List, add_messages]
    structured_output: Optional[Union[ResearchResult, ReasoningResult]]
