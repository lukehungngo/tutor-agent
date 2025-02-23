from typing import Any, List
from langchain.tools import Tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from models.state import State
from multi_agent.reasoner.deep_reasoner import DeepReasoner

class ReasonerTeam:
    def __init__(self, llm: Any, tools: List[Tool] = []):
        self.llm = llm
        self.name = "reasoner_team"
        self.reasoner_agent = DeepReasoner(llm, tools)

    def create_workflow(self) -> CompiledStateGraph:
        workflow = StateGraph(State)
        workflow.add_node("reasoner_agent", self.reasoner_agent.call_model)
        workflow.add_edge("reasoner_agent", END)
        workflow.set_entry_point("reasoner_agent")
        return workflow.compile(name=self.name)

