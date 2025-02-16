from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class UserInput(BaseModel):
    messages: Annotated[list, add_messages]

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    task: Annotated[Optional[str], add_messages]
    answer: Annotated[Optional[str], add_messages]