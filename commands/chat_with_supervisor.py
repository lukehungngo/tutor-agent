from langchain.globals import set_debug
from config import *
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain.tools import tool
from langchain.chains import LLMMathChain
from langgraph_supervisor import create_supervisor
from langgraph.graph.state import CompiledStateGraph
from typing import Any
from multi_agent import *

# set_debug(True)

math_team = MathTeam(settings.open_api_client).create_workflow()
top_supervisor = TopSupervisor(settings.open_api_client).create([math_team])

# Create initial state with the math question
initial_state = {
    "messages": [
        {
            "role": "user",
            "content": "1+2 = ?",
        }
    ]
}

print("Initial state:", initial_state)
result = top_supervisor.invoke(initial_state)
print("\nFinal state:", result)