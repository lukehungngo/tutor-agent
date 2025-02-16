from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from services.stateful_chatbot import StatefulChatbot
from retrievers import tavily_tool
from models.state import ResearchState
from config import settings
from langgraph.prebuilt import ToolNode, tools_condition

class StateManager:
    """Manages the state and graph configuration for the chatbot system."""

    def __init__(self):
        self.memory = MemorySaver()
        self.graph_builder = StateGraph(ResearchState)
        self._configure_graph()

    def _configure_graph(self):
        """Configure the graph with nodes and edges."""
        llm = settings.llm

        tools = [tavily_tool]
        llm = llm.bind_tools(tools=tools)
        stateful_chatbot = StatefulChatbot(llm)
        tools = ToolNode(tools)
        # Define entry and end points
        self.graph_builder.add_node("chatbot", stateful_chatbot.chat)
        self.graph_builder.add_node("tools", tools)
        self.graph_builder.add_conditional_edges(
                "chatbot",
                tools_condition,
        )
        self.graph_builder.set_entry_point("chatbot")
        self.graph_builder.add_edge("tools", "chatbot")


    def get_graph(self):
        """Return the compiled graph with memory checkpointing."""
        return self.graph_builder.compile(checkpointer=self.memory)