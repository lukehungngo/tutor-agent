from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from models.state import State

class PlannerSubtopic(BaseModel):
    """Schema for an individual subtopic"""

    title: str = Field(
        ...
    )  # example="What are the fundamental principles of photosynthesis?"
    description: Optional[str] = Field(
        None
    )  # example="This subtopic covers the basic principles of photosynthesis, including the role of chlorophyll and the process of converting light energy into chemical energy."
    category: Optional[str] = Field(None)  # Beginner, Intermediate, Advanced
    relevance_score: Optional[float] = Field(
        None
    )  # AI-generated importance score (0-1)
    sources: Optional[List[str]] = Field(None)  # Research sources
    recommended_resources: Optional[List[str]] = Field(
        None
    )  # Links to further learning


class PlannerResult(BaseModel):
    """Schema for structured subtopics"""

    task: str = Field(...)  #  example="How does photosynthesis work?"
    subtopics: List[PlannerSubtopic]  # List of subtopics with metadata

class Planner:
    """Plans the execution path between Reasoner and Researcher."""
    
    def __init__(self, llm):
        """Initialize the Planner with an LLM."""
        self.llm = llm

    def plan(self, state: State) -> State:
        """
        Analyze the query and plan whether to use reasoning or research.
        
        Args:
            state: Current state containing messages
            
        Returns:
            Updated state with plan
        """
        messages = state.get("messages", [])
        if not messages:
            return state
            
        query = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        
        # Analyze query to determine if it needs external information
        needs_research = any([
            "latest" in query.lower(),
            "current" in query.lower(),
            "find" in query.lower(),
            "search" in query.lower(),
            "look up" in query.lower(),
            "example of" in query.lower(),
            "real world" in query.lower(),
            "statistics" in query.lower(),
            "data" in query.lower(),
        ])
        
        # Update state with plan
        plan = "research" if needs_research else "reasoning"
        messages.append({
            "role": "system", 
            "content": f"Based on query analysis, using {plan} path.", 
            "plan": plan
        })
        
        return {"messages": messages}

    @staticmethod
    def get_next_step(state: State) -> str:
        """
        Determine next node based on plan.
        
        Args:
            state: Current state containing messages and plan
            
        Returns:
            Name of next node to route to
        """
        messages = state.get("messages", [])
        if not messages:
            return "researcher"
            
        last_message = messages[-1]
        if isinstance(last_message, dict):
            plan = last_message.get("plan")
            if plan == "reasoning":
                return "reasoner"
            return "researcher"
        return "researcher"
