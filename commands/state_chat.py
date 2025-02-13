from services.chatbot import graph
from typing import Any
from langgraph.types import Command

def main():
    user_input = (
    "Can you look up when LangGraph was released? "
    "When you have the answer, use the human_assistance tool for review."
    )
    config : Any = {"configurable": {"thread_id": "1"}}

    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    snapshot = graph.get_state(config)
    print(">>>> VALUES", {k: v for k, v in snapshot.values.items() if k in ("name", "birthday")})
    print(">>>> NEXT", snapshot.next)
    human_command = Command(
        resume={
            "correct": "y",
        },
    )
    events = graph.stream(human_command, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    snapshot = graph.get_state(config)
    print(">>>> VALUES", {k: v for k, v in snapshot.values.items() if k in ("name", "birthday")})
    print(">>>> NEXT", snapshot.next)
    graph.update_state(config, {"name": "LangGraph (library)"})
    snapshot = graph.get_state(config)
    print(">>>> VALUES", {k: v for k, v in snapshot.values.items() if k in ("name", "birthday")})
    print(">>>> NEXT", snapshot.next)
    to_replay = None
    for state in graph.get_state_history(config):
        print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
        print("-" * 80)
        if len(state.values["messages"]) == 3:
            # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
            to_replay = state
    print(to_replay.next if to_replay else "No state to replay")
    print(to_replay.config if to_replay else "No config to replay")
    if to_replay:
        # The `checkpoint_id` in the `to_replay.config` corresponds to a state we've persisted to our checkpointer.
        for event in graph.stream(None, to_replay.config, stream_mode="values"):
            if "messages" in event:
                event["messages"][-1].pretty_print()

if __name__ == "__main__":
    main()