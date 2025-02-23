from typing import Any, List
from langchain.tools import Tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from models.state import State
from langchain import hub
from multi_agent.math.quick_math import QuickMathAgent
from multi_agent.math.deep_math import DeepMathAgent
from langchain.globals import set_debug
import json
SUPERVISOR_PROMPT = """
You are a math supervisor. You pick the best agent to answer the question.

QUESTION/TASK:
{query}

If the question is a simple math question or a math question that can be solved by a calculator, use the quick math agent.
If the question is a complex math question or required a deep understanding or explanation of the math, use the deep math agent.

Your response must be in this exact format:
{{"path": "quick_math_agent|deep_math_agent"}}

"""

class MathTeam:
    def __init__(self, llm: Any, tools: List[Tool] = []):
        self.prompt = hub.pull("hwchase17/react")
        self.llm = llm
        self.quick_math_agent = QuickMathAgent(llm, tools)
        self.deep_math_agent = DeepMathAgent(llm, tools)
        self.name = "math_team"

    def route(self, state: State) -> str:
        messages = state.get("messages", [])
        query = messages[-1].content
        try:
            # Let LLM decide the path
            response = self.llm.invoke(
                SUPERVISOR_PROMPT.format(query=query)
            )
            result = response.content
            result = json.loads(response.content)
            route = result["path"]
            return route
        except Exception as e:
            raise Exception(f"Error processing messages: {str(e)}")
        
    def create_workflow(self) -> CompiledStateGraph:
        workflow = StateGraph(State)
        workflow.add_node("quick_math_agent", self.quick_math_agent.call_model)
        workflow.add_node("deep_math_agent", self.deep_math_agent.call_model)
        workflow.set_conditional_entry_point(self.route, {
            "quick_math_agent": "quick_math_agent",
            "deep_math_agent": "deep_math_agent"
        })
        workflow.add_edge("quick_math_agent", END)
        workflow.add_edge("deep_math_agent", END)
        return workflow.compile(name=self.name)

