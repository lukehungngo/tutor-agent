from langgraph_supervisor import create_supervisor
from config import settings
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import AgentExecutor, create_react_agent
from models.state import State
from langchain.prompts import PromptTemplate
import json
from langchain import hub
from langchain.agents import Tool

prompt = """
You are a helpful assistant that can search the web for information and gather as much information as possible.
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, you should use router tool to choose subgraph and others [{tool_names}] except router tool if needed
Action Input: the input to the action
Observation: the result of the action, decide if you need to continue or not
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
        # self.prompt = PromptTemplate(template=prompt)
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
- "reasoner_team": For logic problems, theoretical concepts, or anything solvable with pre-existing knowledge
- "researcher_team": For current information, real-world examples, or facts needing verification

Your response must be in this exact format:
{{"path": "math_team|reasoner_team|researcher_team", "explanation": "Brief explanation of choice"}}"""

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


class Router:
    def __init__(self, llm):
        self.llm = llm

    def route(self, query: str) -> str:
        try:
            # Let LLM decide the path
            response = self.llm.invoke(ROUTING_PROMPT.format(query=query))
            result = response.content
            result = json.loads(response.content)
            route = result["path"]
            print(f">>>>>> Route: \n Query: {query} \n Result: {result} \n----")
            return route
        except Exception as e:
            print(f"Error processing messages: {str(e)}")
            return "researcher_team"

class TopSupervisor:
    def __init__(self, llm, tools=None, max_iterations=5):
        if tools is None:
            tools = []
        self.llm = llm
        self.tools = tools
        # self.prompt = hub.pull("hwchase17/react")
        self.prompt = PromptTemplate.from_template(prompt)
        self.router = Router(self.llm)
        self.router_tool = Tool(
            name="router",
            description="A tool to route the query to the appropriate agent. Use this first to determine which specialized agent should handle the query.",
            func=self.router.route,
        )
        self.tools.append(self.router_tool)
        self.name = "top_supervisor"
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=max_iterations,
            early_stopping_method="force",  # Force stop when Final Answer is provided
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
