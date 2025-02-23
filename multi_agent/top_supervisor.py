from langgraph_supervisor import create_supervisor
from config import settings
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import AgentExecutor, create_react_agent
from models.state import State
from langchain.prompts import PromptTemplate
import json
from langchain import hub

SUPERVISOR_PROMPT = """You are a high-level supervisor coordinating specialized teams to solve complex problems efficiently, including managing multi-team collaborations when necessary.
Your role:
- Analyze the user's query and determine the most appropriate team(s) to handle it
- Direct tasks to these specialized teams:
  1. Math Team: For calculations, equations, mathematical proofs, and numerical problems
  2. Reasoning Team: For logic problems, deductive reasoning, and analytical challenges
  3. Research Team: For fact-finding, information gathering, and knowledge-based queries

Team Selection Guidelines:
- Math Team: Choose when the query involves numbers, calculations, or mathematical concepts
- Reasoning Team: Choose when the query requires logical analysis or step-by-step problem solving
- Research Team: Choose when the query requires factual information or domain knowledge

Multi-Team Collaboration:
For complex queries that require multiple expertise areas, you can combine teams:
- Math + Reasoning: For problems requiring both calculation and logical analysis
- Research + Math: For problems needing both factual context and mathematical solutions
- Research + Reasoning: For analysis that requires both background knowledge and logical deduction
- All Teams: For complex problems requiring comprehensive analysis

To delegate work to a team, use their respective handoff tools:
- Use the transfer_to_math_team tool for mathematical problems
- Use the transfer_to_reasoner_team tool for reasoning tasks
- Use the transfer_to_researcher_team tool for research queries

When using a handoff tool:
1. First acknowledge the task: "I'll delegate this to the [team] team."
2. Then use the appropriate transfer tool
3. The question will automatically be passed to the team in the messages

For example:
1. For a math problem:
   "I'll delegate this to the math team."
   [Use transfer_to_math_team tool]

2. For a complex problem needing both math and reasoning:
   "I'll first get the calculation from the math team."
   [Use transfer_to_math_team tool]
   [Wait for response]
   "Now I'll have the reasoning team explain the concept."
   [Use transfer_to_reasoner_team tool]

Remember to:
1. Always acknowledge what you're doing before using a handoff tool
2. Use the exact tool names (transfer_to_math_team, etc.)
3. Wait for each team's response before proceeding if sequential work is needed
"""

ROUTING_PROMPT = """You are a routing agent that must decide whether a question/task requires external research, mathematical computation, or can be answered through logical reasoning alone.

QUESTION/TASK:
{query}

First, identify the type of question:
1. Basic Computation: Simple arithmetic, mathematical operations (e.g., "1+2", "what is 15% of 200")
2. Logical Reasoning: Problems solvable with pre-existing knowledge (e.g., "how do loops work", "explain binary search")
3. Information Seeking: Requires external data or verification (e.g., "latest AI developments", "who won the 2024 Super Bowl")

Guidelines:
- Basic computation and pure mathematical questions should use "math"
- Complex mathematical problems requiring explanation should use both "math" and "reasoning"
- Theoretical/conceptual questions should use "reasoning"
- Questions about current events, specific products, or real-world data should use "research"
- If unsure, check if the answer would be the same 6 months ago - if yes, use "reasoning"

Choose ONE path:
- "math_team": For pure calculations and mathematical operations
- "reasoning_team": For logic problems, theoretical concepts, or anything solvable with pre-existing knowledge
- "researcher_team": For current information, real-world examples, or facts needing verification

Your response must be in this exact format:
{{"path": "math_team|reasoning_team|researcher_team", "explanation": "Brief explanation of choice"}}"""

# class TopSupervisor:
#     """Routes the execution path between Reasoner and Researcher."""

#     def __init__(self, llm):
#         """Initialize the Router with an LLM."""
#         self.llm = settings.google_gemini_client
#         self.name = "top_supervisor"

#     def create(self, agents: list[CompiledStateGraph]) -> CompiledStateGraph:
#         workflow = create_supervisor(
#             agents,
#             model=self.llm,
#             prompt=SUPERVISOR_PROMPT,
#             output_mode="last_message",
#             add_handoff_back_messages=True,
#         )
#         return workflow.compile()

class TopSupervisor:
    def __init__(self, llm, tools, max_iterations=10):
        self.llm = llm
        self.tools = tools
        self.prompt = hub.pull("hwchase17/react")
        self.name = "top_supervisor"
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=max_iterations,
            return_intermediate_steps=True,  # Ensure we get all intermediate steps
        )
    def call_model(self, state: State):
        """
        Process messages in the state using the research agent.

        Args:
            state: The current state containing messages and other data

        Returns:
            Dict containing both the processed messages, structured output and raw search results

        Raises:
            ValueError: If messages are invalid
            Exception: If research processing fails
        """
        try:
            messages = state.get("messages", [])
            if not messages:
                raise ValueError("No messages found in state")

            # Get the last message content for research
            last_message = (
                messages[-1].content
                if hasattr(messages[-1], "content")
                else messages[-1]
            )

            # Execute research
            response = self.agent_executor.invoke(
                {"input": last_message, "agent_scratchpad": ""}
            )

            return {
                "messages": [
                    {
                            "role": "assistant",
                            "content": str(response["output"]),
                        }
                ],
            }

        except Exception as e:
            print(f"Error during research: {str(e)}")
            return {
                "messages": [{"role": "assistant", "content": str(e)}],
            }
        
    def route(self, state: State) -> str:
        messages = state.get("messages", [])
        query = messages[-1].content
        try:
            # Let LLM decide the path
            response = self.llm.invoke(
                SUPERVISOR_PROMPT.format(query=query)
            )
            result = response.content
            result = json.loads(response.content)
            route = result["path"]
            return route
        except Exception as e:
            print(f"Error processing messages: {str(e)}")
            return "researcher_team"
    