from multi_agent.state_manager import StateManager
from utils import *


def main(*args):
    graph = StateManager().get_graph()
    generate_graph(graph)


if __name__ == "__main__":
    main()
