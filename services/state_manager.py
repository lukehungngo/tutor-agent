from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from services.researcher import Researcher
from retrievers import tavily_tool, duckduckgo_tool, google_tool, wikipedia_summary_tool, arxiv_tool
from models.state import ResearchState
from services.reflector_with_structured_output import StructuredOutputReflector
from config import settings
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END

class StateManager:
    """Manages the state and graph configuration for the chatbot system."""

    def __init__(self):
        self.memory = MemorySaver()
        self.graph_builder = StateGraph(ResearchState)
        self._configure_graph()

    def _configure_graph(self):
        """Configure the graph with nodes and edges."""
        llm = settings.llm

        tools = [google_tool, wikipedia_summary_tool, duckduckgo_tool, arxiv_tool, tavily_tool]
        researcher = Researcher(settings.open_api_client, tools)
        reflector = StructuredOutputReflector(settings.open_api_client)
        # Define entry and end points
        self.graph_builder.add_node("researcher", researcher.research)
        self.graph_builder.add_node("reflector", reflector.run)
        # Add edges
        self.graph_builder.add_edge("researcher", "reflector")
        self.graph_builder.add_edge("reflector", END)
        # Set entry point
        self.graph_builder.set_entry_point("researcher")
        # Compile graph with memory checkpointing
        self.graph = self.graph_builder.compile(checkpointer=self.memory)
        
    def get_graph(self):
        """Return the compiled graph with memory checkpointing."""
        return self.graph