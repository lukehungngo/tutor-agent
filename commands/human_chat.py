from services.chatbot import graph
from typing import Any
from langgraph.types import Command

def main():
    config : Any = {"configurable": {"thread_id": "1"}}
    user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()
	
    snapshot = graph.get_state(config)
    print(">>>> NEXT", snapshot.next)
        
    human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
    )

    human_command = Command(resume={"data": human_response})

    events = graph.stream(human_command, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

if __name__ == "__main__":
    main()