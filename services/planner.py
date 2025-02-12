from config import *
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.topic import Subtopics

# Define Output Parser (Enforces structured JSON)
parser = PydanticOutputParser(pydantic_object=Subtopics)

# Define AI Prompt Template
prompt = PromptTemplate(
    template="""
    You are an AI that generates structured learning subtopics.

    Task: {task}
    Additional Context: {data}

    Existing Subtopics: {subtopics}

    Generate up to {max_subtopics} additional subtopics.
    {format_instructions}
    """,
    input_variables=["task", "data", "subtopics", "max_subtopics"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Create AI Pipeline (Prompt → LLM → Parser)
chain = prompt | settings.open_api_client | parser

async def construct_subtopics(task: str, data: str, max_subtopics: int = 5, subtopics: list = []) -> list:
    """
    Constructs structured subtopics based on a given learning task.

    Args:
        task (str): The main learning topic.
        data (str): Additional contextual data.
        config: Configuration settings (e.g., max subtopics).
        subtopics (list, optional): Existing subtopics (if any). Defaults to [].

    Returns:
        list: A structured list of generated subtopics.
    """
    try:
        print(f"\n🤖 Calling {settings.open_api_model} model...\n")

        output = chain.invoke({
            "task": task,
            "data": data,
            "subtopics": subtopics,
            "max_subtopics": max_subtopics
        })
        return output.subtopics  # Return structured list of subtopics

    except Exception as e:
        print("Exception in parsing subtopics:", e)
        return subtopics  # Fallback to existing subtopics in case of failure