from langchain.globals import set_debug
from config import *
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain.tools import tool
from langchain.chains import LLMMathChain
from langgraph_supervisor import create_supervisor
from langgraph.graph.state import CompiledStateGraph
from typing import Any
from utils import *

set_debug(True)


# @tool
# def math_tool(x: int, y: int) -> int:
#     """Calculate the sum of two numbers."""
#     print(f"Calculating {x} + {y}")
#     return x + y

# PROMPT = """
# You are a math expert.
# """
# math_agent : Any = create_react_agent(
#     model=settings.google_gemini_client,
#     tools=[math_tool],
#     prompt=PROMPT,
#     name="math_expert"
# )

# SUPERVISOR_PROMPT = """
# "You are a team supervisor managing a math expert."
# "For math problems, use math_expert agent."
# """
# workflow = create_supervisor(
#   [math_agent],
#   model=settings.open_api_client,
#   prompt=SUPERVISOR_PROMPT,
#   output_mode="last_message",
# )

# top_supervisor : Any = workflow.compile()
# generate_graph(top_supervisor)
# event = top_supervisor.invoke(
#     {"messages": [{"role": "user", "content": "1+2=?"}]},
# )
# for e in event['messages']:
#     e.pretty_print()
# for e in event:
#     print("Type of e: ", type(e))
#     print(e)

# Create specialized agents


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )


model = settings.open_api_client
math_agent: Any = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time.",
)

research_agent: Any = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math.",
)

workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    ),
)
app = workflow.compile()
result = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "what's the combined headcount of the FAANG companies in 2024?",
            }
        ]
    }
)
