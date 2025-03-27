from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import json
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BloomLevel(Enum):
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


@dataclass
class Question:
    question: str
    bloom_level: BloomLevel
    context: str
    hint: Optional[str] = None
    answer: Optional[str] = None


class ExamGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.prompt_template = self._create_aggregated_prompt()

    def _create_aggregated_prompt(self):
        """Create a single prompt template that handles all Bloom's Taxonomy levels."""
        template = """Given the following content:
{context}

Generate questions based on Bloom's Taxonomy levels according to these requirements:
{level_requirements}

For each question, follow this format:
Level: <bloom_level>
Question: <question>
Suggested Answer: <answer>
Hint: <hint>

Guidelines for each level:
- REMEMBER: Test recall of basic facts, definitions, concepts, or principles
- UNDERSTAND: Require explaining ideas or concepts in their own words
- APPLY: Require applying learned information to solve problems or new situations
- ANALYZE: Require breaking down information and establishing relationships between concepts
- EVALUATE: Require making judgments or evaluating outcomes based on criteria
- CREATE: Require creating something new or proposing alternative solutions

Return the questions in JSON format like this:
{{
    "questions": [
        {{
            "level": "remember",
            "question": "...",
            "hint": "...",
            "answer": "..."
        }},
        ...
    ]
}}"""

        return PromptTemplate(
            input_variables=["context", "level_requirements"], template=template
        )

    def _format_level_requirements(self, questions_per_level: Dict[BloomLevel, int]):
        """Format the requirements for each Bloom's level."""
        requirements = []
        for level, count in questions_per_level.items():
            if count > 0:
                requirements.append(
                    f"- Generate {count} question(s) for {level.value.upper()} level"
                )
        return "\n".join(requirements)

    async def generate_exam(
        self, chunks: List[str], questions_per_level: Dict[BloomLevel, int]
    ) -> List[Question]:
        """Generate an exam with specified number of questions per Bloom's level."""
        questions = []

        for chunk in chunks:
            try:
                # Format level requirements
                level_requirements = self._format_level_requirements(
                    questions_per_level
                )

                # Create and run the chain
                chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
                response = await chain.arun(
                    context=chunk, level_requirements=level_requirements
                )

                # Parse the JSON response
                try:
                    result = json.loads(response)
                    for q in result["questions"]:
                        questions.append(
                            Question(
                                question=q["question"],
                                bloom_level=BloomLevel(q["level"]),
                                context=chunk,
                                answer=q["answer"],
                                hint=q["hint"],
                            )
                        )
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse response as JSON: {str(e)}")
                    logger.debug(f"Response was: {response}")
                    continue

            except Exception as e:
                logger.error(f"Failed to generate questions: {str(e)}")
                continue

        return questions
