from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from services.state_manager import StateManager
from models.state import State
from typing import Any

router = APIRouter()
state_manager = StateManager()
graph = state_manager.get_graph()

@router.post("/api/chat")
async def chat(id: str, message: str):
    config: Any = {"configurable": {"thread_id": id}}
    events = list(graph.stream(
        {"messages": [{"role": "user", "content": message}]},
        config,
        stream_mode="values",
    ))
    # Get the last message from the final event
    # Log all messages from events
    for event in events:
        event["messages"][-1].pretty_print()
    final_message = events[-1]["messages"][-1].content
    return {"response": final_message}