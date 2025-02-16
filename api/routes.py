from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from services.state_manager import StateManager
from models.state import State, ReasoningResult, ResearchResult
from typing import Any
from pydantic import BaseModel
from uuid import uuid4
from datetime import datetime

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
    events = list(
        graph.stream(
            {"messages": [{"role": "user", "content": request.message}]},
            config,
            stream_mode="values",
        )
    )
    last_event = events[-1]
    structured_output = last_event.get("structured_output", None)
    response = {
        "response": {
            "content": last_event["messages"][-1].content,
            "id": request.id,
            "timestamp": datetime.now(),
        }
    }
    if structured_output and (isinstance(structured_output, ReasoningResult) or isinstance(structured_output, ResearchResult)):
        uuid = uuid4()
        response["response"].update(
            {
                "structured_output": {},
                "uuid": uuid,
            }
        )
        if isinstance(structured_output, ReasoningResult):
            response["response"]["structured_output"]["reasoning_result"] = structured_output.model_dump()
        elif isinstance(structured_output, ResearchResult):
            response["response"]["structured_output"]["research_result"] = structured_output.model_dump()

    return response
