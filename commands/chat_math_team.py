from multi_agent.math.math_expert import MathTeam
from config import settings
from models.state import State

# Initialize the math team
math_team = MathTeam(settings.open_api_client)
# user_input = "What is the square root of 16?"
user_input = """Evaluate this expression: numexpr.evaluate("sin(45 * pi/180) + cos(30 * pi/180)")"""
event = math_team.call_model({"messages": [{"role": "user", "content": user_input}]})

last_event = None
for e in event:
    print("Type of e: ", type(e))
    print(e)
    last_event = e
if last_event:
    print("-------------------------------- Type of last_event")
    print(type(last_event))
    print(last_event)
