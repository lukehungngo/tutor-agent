from langchain.globals import set_debug
from config import *
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain.tools import tool
from langchain.chains import LLMMathChain
from langgraph_supervisor import create_supervisor
from langgraph.graph.state import CompiledStateGraph
from typing import Any
from multi_agent import *
from multi_agent.state_manager import *

# set_debug(True)

state_manager = StateManager()
graph = state_manager.get_graph()

config: Any = {"configurable": {"thread_id": "123"}}

user_input = """Evaluate this expression: numexpr.evaluate("sin(45 * pi/180) + cos(30 * pi/180)")"""
user_require_explanation = """Explain how to solve the following expression: numexpr.evaluate("sin(45 * pi/180) + cos(30 * pi/180)")"""
event = graph.invoke(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
print(event)

# print("--------------------------------")
# event = graph.invoke({"messages": [{"role": "user", "content": user_require_explanation}]})
# print(event)
