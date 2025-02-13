from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from services.state_manager import StateManager
from models.state import State
from typing import Any

router = APIRouter()
state_manager = StateManager()
graph = state_manager.get_graph()

@router.get("/api/chat")
async def chat(message: str):
    config: Any = {"configurable": {"thread_id": "1"}}

    async def stream_response():
        initial_state = State(messages=[{"role": "user", "content": message}])
        async for event in graph.astream(initial_state, config, stream_mode="values"):
            if "messages" in event and event["messages"]:
                yield event["messages"][-1].content + "\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")
