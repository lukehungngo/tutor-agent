from multi_agent import *
from config import settings
from models.state import State

# Initialize the math team
math_team = MathTeam(settings.google_gemini_client)
# user_input = "What is the square root of 16?"
user_input = """Evaluate this expression: numexpr.evaluate("sin(45 * pi/180) + cos(30 * pi/180)")"""
user_require_explanation = """Explain how to solve the following expression: numexpr.evaluate("sin(45 * pi/180) + cos(30 * pi/180)")"""
event = math_team.quick_math_agent.call_model(
    {"messages": [{"role": "user", "content": user_input}]}
)
print("Quick Math Agent")
print(event)

print("Deep Math Agent")
event = math_team.deep_math_agent.call_model(
    {"messages": [{"role": "user", "content": user_input}]}
)
print(event)

print("Math Team")
graph = math_team.create_workflow()
event = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
print(event)

print("--------------------------------")
event = graph.invoke(
    {"messages": [{"role": "user", "content": user_require_explanation}]}
)
print(event)
