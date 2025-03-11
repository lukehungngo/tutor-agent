from multi_agent.state_manager import StateManager
from utils import *
from multi_agent.top_supervisor import TopSupervisor
from config import settings


def main(*args):
    graph = StateManager().get_graph()
    generate_graph(graph)
    top_supervisor = TopSupervisor(settings.google_gemini_client, [])
    generate_graph(top_supervisor)


if __name__ == "__main__":
    main()
