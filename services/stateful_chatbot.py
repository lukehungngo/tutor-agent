from models.state import ResearchState
from typing import Any

class StatefulChatbot:
    def __init__(self, llm):
        """
        Initialize Chatbot with tools.
        
        Args:
            llm: The LLM to use for the chatbot
        """
        self.llm = llm
        self.structured_llm = llm.with_structured_output(ResearchState)
    
    def chat(self, state: ResearchState) -> ResearchState:
        print(">>>>>>>>>>> State:", state)
        try:
            messages = state.get("messages", [])
            return self.structured_llm.invoke(messages)
        except Exception as e:
            raise Exception(f"Error processing messages: {str(e)}")
