from models.state import State
from langgraph.types import Command
import json

ROUTING_PROMPT = """You are a routing agent that must decide whether a question/task requires external research or can be answered through logical reasoning alone.

QUESTION/TASK:
{query}

First, identify the type of question:
1. Basic Computation: Simple arithmetic, mathematical operations (e.g., "1+2", "what is 15% of 200")
2. Logical Reasoning: Problems solvable with pre-existing knowledge (e.g., "how do loops work", "explain binary search")
3. Information Seeking: Requires external data or verification (e.g., "latest AI developments", "who won the 2024 Super Bowl")

Guidelines:
- Basic computation and mathematical questions should ALWAYS use "reasoning"
- Theoretical/conceptual questions should use "reasoning"
- Questions about current events, specific products, or real-world data should use "research"
- If unsure, check if the answer would be the same 6 months ago - if yes, use "reasoning"

Choose ONE path:
- "deep_reasoner": For computations, logic problems, theoretical concepts, or anything solvable with pre-existing knowledge
- "researcher": For current information, real-world examples, or facts needing verification

Your response must be in this exact format:
{{"path": "deep_reasoner|researcher", "explanation": "Brief explanation of choice"}}"""

class Router:
    """Routes the execution path between Reasoner and Researcher."""
    
    def __init__(self, llm):
        """Initialize the Router with an LLM."""
        self.llm = llm

    def route(self, state: State) -> str:
        """
        Use LLM to analyze the query and route whether to use reasoning or research.
        
        Args:
            state: Current state containing messages
            
        Returns:
            Updated state with route
        """
        messages = state.get("messages", [])
        query = messages[-1].content
        
        try:
            # Let LLM decide the path
            response = self.llm.invoke(
                ROUTING_PROMPT.format(query=query)
            )
            result = response.content
            result = json.loads(response.content)
            route = result["path"]
            explanation = result["explanation"]
            print(f"Route: {route}, Explanation: {explanation}")
            return route
        
        except Exception as e:
            print(f"Error routing: {e}")
            return "researcher"