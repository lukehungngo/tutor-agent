"""
Command to test the planner's subtopic generation.
"""
import asyncio
from services.planner import construct_subtopics

def main(*args):
    asyncio.run(test_generate_subqueries())

async def test_generate_subqueries():
    # Test input
    task = "I am a junior software engineer and I want to learn about backend software development."
    data = ""
    # Call the function
    result = await construct_subtopics(task, data)
    
    # Write results to JSON file
    import json
    with open('subtopics.json', 'w') as f:
        json.dump([subtopic.model_dump() for subtopic in result], f, indent=4)