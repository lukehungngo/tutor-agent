from langchain import hub
from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from retrievers import *
from utils import *
from config import settings
from pydantic import BaseModel
from typing import Optional, Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.globals import set_debug
from langgraph.types import Command
# Enable debug mode
# set_debug(True)

# --- React Agent ---
prompt = hub.pull("hwchase17/react")
""" Prompt template for the react agent
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

tools = [tavily_tool, duckduckgo_tool]

# Create the react agent
agent = create_react_agent(settings.open_api_client, tools, prompt)
# Create the executor with handle_parsing_errors enabled

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations = 10
)
# Invoke the agent with the required variables
# result = agent_executor.invoke({
#     "input": "I am a junior software engineer, I want to learn about AI applications, not just the theory, but also the practical applications, what courses should I take?",
#     "agent_scratchpad": "", # begin with empty scratchpad
# }, config={"configurable": {"thread_id": "1"}})
# print("Type of React Agent result: ", type(result))
# print("React Agent result: ", result)

# --- Structured Output Agent ---
class SubTopic(BaseModel):
    title: str
    description: Optional[str] = None
    url: Optional[str] = None

class ResponseSchema(BaseModel):
    summary: str
    sub_topics: Optional[list[SubTopic]] = None

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    structured_output: Optional[ResponseSchema]


structured_llm = settings.open_api_client.with_structured_output(ResponseSchema)

# response = structured_llm.invoke(result["output"])
# print("Type of Structured Output Agent result: ", type(response))
# print("Structured Output Agent result: ", response)

# --- Langgraph ---

def researcher(state: AgentState):
    """Runs the researcher agent."""
    try:
        messages = state.get("messages", [])
        response = agent_executor.invoke({"input": messages[-1].content})
        return {"messages": [response["output"]]}
    except Exception as e:
        return {"messages": ["An error occurred during research."]}


def reflect_with_structured_output(state: AgentState):
    """Processes the research result into structured output."""
    try:
        messages = state.get("messages", [])
        last_message_content = messages[-1].content if messages else ""
        structured_response = structured_llm.invoke(last_message_content)
        return Command(
            update={
                "structured_output": structured_response, 
                "messages": [{"role": "assistant", "content": str(structured_response)}],
            },
        )
    except Exception as e:
        print("Error: ", e)
        return Command(
            update={
                "structured_output": None,
                "messages": [{"role": "assistant", "content": "Failed to structure response."}],
            },
        )


def format_response(state: AgentState):
    return {"structured_response": state["messages"][-1].content}

workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher)
workflow.add_node("reflect_with_structured_output", reflect_with_structured_output)
workflow.add_edge("researcher", "reflect_with_structured_output")
workflow.add_edge("reflect_with_structured_output", END)
workflow.set_entry_point("researcher")

graph = workflow.compile()
# generate_graph(graph)
user_input = "I am a junior software engineer, I want to learn about AI applications, not just the theory, but also the practical applications, what courses should I take?"
event = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
)
last_event = None
for e in event:
    print("Type of e: ", type(e))
    print(e)
    last_event = e
if last_event:
    print("-------------------------------- Type of last_event")
    print(type(last_event))
    # print(last_event)
    print("-------------------------------- Structured Output")
    # Access structured_output
    structured_output = last_event["reflect_with_structured_output"]["structured_output"]
    print(structured_output)
    print("-------------------------------- Messages")
    # Access messages
    messages = last_event["reflect_with_structured_output"]["messages"]
    print( messages)