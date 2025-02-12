from services.planner import construct_subtopics
from models.topic import Subtopics

async def test_generate_subqueries():
    # Test input
    task = "I am a junior software engineer and I want to learn about the latest trends in software development."
    data = "Focus on backend development."
    # Call the function
    result = await construct_subtopics(task, data)
    
    # Write results to JSON file
    import json
    with open('subtopics.json', 'w') as f:
        json.dump([subtopic.model_dump() for subtopic in result], f, indent=4)