from multi_agent.state_manager import StateManager
from typing import Any


def main():
    state_manager = StateManager()
    graph = state_manager.get_graph()
    config: Any = {"configurable": {"thread_id": "1"}}

    def stream_graph_updates(user_input: str):
        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode="values",
        )
        for event in events:
            event["messages"][-1].pretty_print()

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q", ""]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break


if __name__ == "__main__":
    main()
