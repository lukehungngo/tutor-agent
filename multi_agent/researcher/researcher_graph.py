from typing import Any, List
from langchain.tools import Tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from models.state import State
from multi_agent.researcher.researcher import Researcher
from tools.retrievers import (
    tavily_tool,
    duckduckgo_tool,
    google_tool,
    wikipedia_summary_tool,
    arxiv_tool,
)


class ResearcherTeam:
    def __init__(self, llm: Any, tools: List[Tool] = []):
        default_tools = [
            google_tool,
            wikipedia_summary_tool,
            duckduckgo_tool,
            arxiv_tool,
            tavily_tool,
        ]
        self.llm = llm
        self.name = "researcher_team"
        self.tools = tools if tools else default_tools
        self.researcher_agent = Researcher(llm, self.tools)

    def create_workflow(self) -> CompiledStateGraph:
        workflow = StateGraph(State)
        workflow.add_node("researcher_agent", self.researcher_agent.call_model)
        workflow.add_edge("researcher_agent", END)
        workflow.set_entry_point("researcher_agent")
        return workflow.compile(name=self.name)
