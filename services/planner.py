from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.topic import Subtopic
from models.state import State
import json

class Planner:
    def __init__(self, llm):
        """
        Initialize Planner with LLM and configure parser/prompt.

        Args:
            llm: The LLM to use for planning
        """
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=List[Subtopic])

        self.prompt = PromptTemplate(
            template="""
            You are an AI that generates structured learning subtopics.

            Task: {task}
            Description: {description}

            Generate up to {max_subtopics} subtopics.
            Each subtopic should have a title and an optional description, category,
            relevance_score, sources, and recommended_resources.

            Format Instructions:
            {format_instructions}
            """,
            input_variables=["task", "description", "max_subtopics"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        self.chain = self.prompt | self.llm | self.parser
    
    def plan(self, state: State) -> Dict[str, List[Subtopic]]:
        """
        Process the state to generate subtopics using the LLM.

        Args:
            state: The current state containing task and context data

        Returns:
            Dict containing the processed messages and generated subtopics

        Raises:
            Exception: If planning or parsing fails
        """
        try:
            messages = state["messages"]
            # Extract task and description from the latest user message
            if messages and isinstance(messages[-1], HumanMessage):
                user_message = messages[-1].content
            else:
                raise ValueError("No user message found in state")

            # Extract the task and description from the user message using a prompt
            extraction_prompt = PromptTemplate.from_template(
                """You are an expert at extracting task and description.
                Extract the task, description, and maximum subtopics from this user message: {user_message}.
                If the user does not specify the maximum number of subtopics then do the deduction or default to 5.
                Return a JSON consisting of "task", "description", and "max_subtopics".

                The value of the return should be a proper string, and cannot contain new line."""
            )
            extraction_chain = extraction_prompt | self.llm

            task_info_str = extraction_chain.invoke({"user_message": user_message})

            try:
                task_info = json.loads(task_info_str)
                task = task_info.get("task")
                description = task_info.get("description")
                max_subtopics = int(task_info.get("max_subtopics", 5)) #default is 5
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Error decoding JSON or accessing keys: {e}")
                task = user_message
                description = ""
                max_subtopics = 5

            output = self.chain.invoke({
                "task": task,
                "description": description,
                "max_subtopics": max_subtopics
            })

            # Update state with new subtopics
            return {"topic": output}

        except Exception as e:
            raise Exception(f"Error in planning: {str(e)}")