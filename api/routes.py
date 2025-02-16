from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from services.state_manager import StateManager
from models.state import State, ResearchState
from typing import Any
from pydantic import BaseModel

router = APIRouter()
state_manager = StateManager()
graph = state_manager.get_graph()

class ChatRequest(BaseModel):
    id: str
    message: str

@router.post("/api/chat")
async def chat(request: ChatRequest):
    print(request)
    config: Any = {"configurable": {"thread_id": request.id}}
    events = list(graph.stream(
        {"messages": [{"role": "user", "content": request.message}], "task": None, "answer": None},
        config,
        stream_mode="values",
    ))
    # Get the last message from the final event
    # Log all messages from events
    for event in events:
        event["task"].pretty_print()
    final_message = events[-1]["task"].answer
    return {"response": final_message}