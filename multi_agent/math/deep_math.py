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
from markdown_text_clean import clean_text

PROMPT = """You are a math expert who can solve mathematical problems.
Question:
{question}

If the given problem is simple, solve it directly and return human readable answer.
If the given problem is complex, break it down into smaller problems and solve each one, and provide formula and the result for each step.
If user asks for a formula, return the formula.
If user asks for a calculation, return the calculation.
If user asks for a definition, return the definition.
If user asks for a theorem, return the theorem.
If user asks for a proof, return the proof.
If user asks for a formula, return the formula.
If user asks for a calculation, return the calculation.
If user asks for a definition, return the definition.

RESPONSE TYPES:
- For calculations: Show both the formula and the step-by-step solution.
- For definitions: Provide a clear explanation with examples.
- For theorems: State the theorem and explain its significance.
- For proofs: Break down into logical steps.
- For formulas: Show the formula, explain variables, and provide an example.

YOUR RESPONSE SHOULD BE:
A natural, flowing markdown document that reads like an expert analysis.

FORMATTING GUIDELINES:
- Use --- for natural topic transitions (when needed).
- Use $ for inline math formulas (e.g. $E = mc^2$).
- Use $$ for block math formulas.
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


class DeepMathAgent:
    def __init__(self, llm: Any, tools: List[Tool] = []):
        self.llm = llm
        self.tools = tools
        self.name = "deep_math_agent"
        self.prompt = PromptTemplate.from_template(template=PROMPT)
        self.llm_math = LLMMathChain.from_llm(self.llm)

    def call_model(self, state: Dict) -> Dict:
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("No messages found in state")
        if isinstance(messages[-1], Dict):
            last_message = messages[-1]["content"]
        else:
            last_message = messages[-1].content
        try:
            # Format the prompt with the last message
            formatted_prompt = self.prompt.format(question=last_message)
            # Invoke LLM with the formatted prompt
            response = self.llm.invoke(formatted_prompt)
            return {
                "messages": [
                    {"role": "assistant", "content": clean_text(response.content)}
                ]
            }
        except Exception as e:
            raise Exception(f"Error processing messages: {str(e)}")
