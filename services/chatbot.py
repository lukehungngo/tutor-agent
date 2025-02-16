from typing import Dict, List
from models.state import State
from langchain_core.messages import BaseMessage


class Chatbot:
    def __init__(self, llm):
        """
        Initialize Chatbot with tools.

        Args:
            llm: The LLM to use for the chatbot
        """
        self.llm = llm

    def chat(self, state: State) -> Dict[str, List[BaseMessage]]:
        """
        Process messages in the state using the LLM.

        Args:
            state: The current state containing messages and other data

        Returns:
            Dict containing the processed messages

        Raises:
            ValueError: If messages are invalid
            Exception: If LLM processing fails
        """
        messages = state.get("messages", [])

        try:
            response = self.llm.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            raise Exception(f"Error processing messages: {str(e)}")
