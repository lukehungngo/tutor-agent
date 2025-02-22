from models.state import State
from langgraph.types import Command
from langgraph_supervisor import create_supervisor
from langgraph.graph import StateGraph
from typing import Any
from config import settings
from multi_agent.math.math_expert import MathTeam
from langgraph.graph.state import CompiledStateGraph

SUPERVISOR_PROMPT = """
"You are a team supervisor managing a math expert."
"For math problems, use math_expert agent."
"""


class TopSupervisorBuilder:
    """Routes the execution path between Reasoner and Researcher."""

    def __init__(self, llm):
        """Initialize the Router with an LLM."""
        self.llm = settings.google_gemini_client
        self.name = "top_supervisor"

    def create(self, agents: list[CompiledStateGraph]) -> CompiledStateGraph:
        workflow = create_supervisor(
            agents,
            model=self.llm,
            prompt=SUPERVISOR_PROMPT,
            output_mode="last_message",
        )
        return workflow.compile()


top_supervisor_builder = TopSupervisorBuilder(settings.google_gemini_client)
math_team = MathTeam(settings.open_api_client)
top_supervisor = top_supervisor_builder.create([])
