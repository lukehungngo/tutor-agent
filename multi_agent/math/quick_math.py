from typing import Any, List, Dict
from langchain.tools import Tool
from langchain.chains.llm_math.base import LLMMathChain


class QuickMathAgent:
    def __init__(self, llm: Any, tools: List[Tool] = []):
        self.llm = llm
        self.tools = tools
        self.name = "quick_math_agent"
        self.llm_math = LLMMathChain.from_llm(self.llm)

    def call_model(self, state: Dict) -> Dict:
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("No messages found in state")
        if isinstance(messages[-1], Dict):
            last_message = messages[-1]["content"]
        else:
            last_message = messages[-1].content
        print(">>>>>>>>>>>>>>>>", last_message)
        try:
            # Format the prompt with the last message
            response = self.llm_math.invoke({"question": last_message})
            # Extract just the answer string from the response
            answer = response["answer"]
            return {"messages": [{"role": "assistant", "content": answer}]}
        except Exception as e:
            raise Exception(f"Error processing messages: {str(e)}")
