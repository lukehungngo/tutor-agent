from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from multi_agent.researcher.researcher_graph import ResearcherTeam
from multi_agent.reasoner.reasoner_graph import ReasonerTeam
from multi_agent.math.math_graph import MathTeam
from models.state import State
from multi_agent.reflector_with_structured_output import StructuredOutputReflector
from config import settings
from langgraph.graph import END
from multi_agent.top_supervisor import TopSupervisor
from langchain.globals import set_debug

# set_debug(True)


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

        researcher_team = ResearcherTeam(settings.open_api_client).create_workflow()
        reasoner_team = ReasonerTeam(settings.open_api_client).create_workflow()
        math_team = MathTeam(settings.open_api_client).create_workflow()

        top_supervisor = TopSupervisor(settings.open_api_client, tools=[])
        reflector = StructuredOutputReflector(
            settings.open_api_client, settings.open_api_client
        )

        self.graph_builder.add_node("top_supervisor", top_supervisor.call_model)
        self.graph_builder.add_node("researcher_team", researcher_team)
        self.graph_builder.add_node("reasoner_team", reasoner_team)
        self.graph_builder.add_node("math_team", math_team)
        self.graph_builder.add_node("reflector", reflector.run)
        # Add edges to end
        self.graph_builder.add_edge("top_supervisor", "reflector")
        self.graph_builder.add_conditional_edges(
            "top_supervisor",
            top_supervisor.router.route,
            {
                "researcher_team": "researcher_team",
                "math_team": "math_team",
                "reasoner_team": "reasoner_team",
            },
        )
        self.graph_builder.add_edge("researcher_team", "top_supervisor")
        self.graph_builder.add_edge("reasoner_team", "top_supervisor")
        self.graph_builder.add_edge("math_team", "top_supervisor")
        self.graph_builder.add_edge("reflector", END)
        # Set entry point
        self.graph_builder.set_entry_point("top_supervisor")

        # Compile graph
        self.graph = self.graph_builder.compile(checkpointer=self.memory)

    def get_graph(self):
        """Return the compiled graph with memory checkpointing."""
        return self.graph
