from langgraph.graph import StateGraph
from typing import TypedDict
from config import settings
from langgraph.graph import END
from typing import Annotated
from langgraph.graph import add_messages
from langchain.globals import set_debug

# Enable debug mode
set_debug(True)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    structured_response: dict


def format_response(state: AgentState):
    return {"structured_response": state["messages"][-1].content}


def chatbot(state: AgentState):
    messages = state.get("messages", [])
    response = settings.DEFAULT_LLM.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(AgentState)
workflow.add_node("llm", chatbot)
workflow.add_node("response_formatter", format_response)
workflow.add_edge("llm", "response_formatter")
workflow.add_edge("response_formatter", END)
workflow.set_entry_point("llm")

graph = workflow.compile()
user_input = "Explain quantum entanglement"
event = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
)
for e in event:
    print("Type of e: ", type(e))
    print(e)
