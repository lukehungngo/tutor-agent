from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from typing import Annotated
from langgraph.types import interrupt, Command
from typing import Any


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


@tool
# Note that because we are generating a ToolMessage for a state update, we
# generally require the ID of the corresponding tool call. We can use
# LangChain's InjectedToolCallId to signal that this argument should not
# be revealed to the model in the tool's schema.
def human_assistance_update_state(
    state: Any, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "state": state,
        },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_state = state
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_state = human_response.get("state", state)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "state": verified_state,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)
