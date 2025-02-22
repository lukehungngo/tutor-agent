from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from multi_agent.researcher import Researcher
from multi_agent.reasoner.deep_reasoner import DeepReasoner
from tools.retrievers import (
    tavily_tool,
    duckduckgo_tool,
    google_tool,
    wikipedia_summary_tool,
    arxiv_tool,
)
from models.state import State
from multi_agent.reflector_with_structured_output import StructuredOutputReflector
from config import settings
from langgraph.graph import END
from langgraph.graph import START
from multi_agent.top_supervisor import top_supervisor


# set_debug(True)
class StateManager:
    """Manages the state and graph configuration for the chatbot system."""

    def __init__(self):
        self.memory = MemorySaver()
        # builder = StateGraph(OverallState,input=InputState,output=OutputState)
        self.graph_builder = StateGraph(State)
        self._configure_graph()

    def _configure_graph(self):
        """Configure the graph with nodes and edges."""
        # Initialize components
        tools = [
            google_tool,
            wikipedia_summary_tool,
            duckduckgo_tool,
            arxiv_tool,
            tavily_tool,
        ]

        router = top_supervisor
        researcher = Researcher(settings.google_gemini_client, tools)
        deep_reasoner = DeepReasoner(
            settings.open_api_client, settings.google_gemini_client
        )
        reflector = StructuredOutputReflector(
            settings.open_api_client, settings.google_gemini_client
        )

        # Add nodes
        self.graph_builder.add_node("researcher", researcher.research)
        self.graph_builder.add_node("deep_reasoner", deep_reasoner.run)
        self.graph_builder.add_node("reflector", reflector.run)

        # Add edges to end
        self.graph_builder.add_edge(
            "researcher", "reflector"
        )  # Research path needs structuring
        self.graph_builder.add_edge("reflector", END)
        self.graph_builder.add_edge(
            "deep_reasoner", END
        )  # Reasoner output is already structured

        # Set entry point
        self.graph_builder.add_conditional_edges(
            START,
            router.route,
            {"researcher": "researcher", "deep_reasoner": "deep_reasoner"},
        )

        # Compile graph
        self.graph = self.graph_builder.compile(checkpointer=self.memory)

    def get_graph(self):
        """Return the compiled graph with memory checkpointing."""
        return self.graph
