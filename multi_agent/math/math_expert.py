from typing import Any, List, Dict
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langgraph.graph import StateGraph, START, END
import numexpr
from models.state import State
from langchain.globals import set_debug
from langchain.chains.llm_math.base import LLMMathChain
import math
# set_debug(True)

PROMPT = """You are a math expert who can solve mathematical problems. 
Input:
{input}

ONLY RETURN THE ANSWER IN HUMAN READABLE LANGUAGE WITH MD FORMAT.

If the given problem is simple, solve it directly and return human readable answer.
If the given problem is complex, break it down into smaller problems and solve each one, and provide formula in MD format and the result for each step.

'''
Examples:

1. Simple Direct Problem:
Question: What is the area of a rectangle with length 12 cm and width 5 cm?
Answer: Area = length * width = 12 * 5 = 60 square centimeters

2. Code Execution:
Question: Evaluate this expression: numexpr.evaluate("sin(45 * pi/180) + cos(30 * pi/180)")
Code Verification: This expression calculates sin(45°) + cos(30°)
Result: 1.5355890141276208

3. Complex Problem Breakdown:
Question: Find the volume and surface area of a cylinder with radius 4 cm and height 10 cm.
Step 1: Calculate the volume
Formula: V = πr²h
Calculation: V = π * 4² * 10 = 502.65 cubic centimeters

Step 2: Calculate the surface area
Formula: A = 2πr² + 2πrh (sum of two circular bases and rectangular lateral surface)
Base area: 2πr² = 2π * 4² = 100.53 square centimeters
Lateral area: 2πrh = 2π * 4 * 10 = 251.33 square centimeters
Total surface area: 351.86 square centimeters
'''
"""

math_tools = [
    Tool(
        name="Calculator",
        func=lambda x: float(numexpr.evaluate(x)),
        description="Evaluates mathematical expressions (e.g., '2 + 2', 'sin(45 * pi/180)'). Input is a string.",
    ),
]


class MathTeam:
    def __init__(self, llm: Any, tools: List[Tool] = math_tools):
        # # self.llm = LLMMathChain.from_llm(llm)
        # self.llm = llm
        # self.tools = tools
        # self.name = "math_expert"
        # self.prompt = PromptTemplate.from_template(template=PROMPT)
        # self.agent = create_react_agent(self.llm, self.tools, prompt=self.prompt)
        # self.agent_executor = AgentExecutor(
        #     agent=self.agent,
        #     tools=self.tools,
        #     verbose=False,
        #     handle_parsing_errors=True,
        #     max_iterations=5,
        #     return_intermediate_steps=True,
        # )
        self.llm = llm
        self.tools = []
        self.name = "math_expert"
        self.prompt = PromptTemplate.from_template(template=PROMPT)

    def call_model(self, state: Dict) -> Dict:
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("No messages found in state")

        last_message = messages[-1]["content"]
        try:
            # Format the prompt with the last message
            formatted_prompt = self.prompt.format(input=last_message)
            print(formatted_prompt)
            # Invoke LLM with the formatted prompt
            response = self.llm.invoke(formatted_prompt)
            print(response)
            return {"messages": [response]}
        except Exception as e:
            raise Exception(f"Error processing messages: {str(e)}")

    def create_workflow(self) -> Any:
        workflow = StateGraph(State)
        workflow.add_node("agent", self.call_model)
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", END)
        return workflow.compile(name=self.name)
