from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from utils import logger
import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pre_trained_model.google_gemma import Gemma3Model
import torch
from typing import Union, Any

BLOOM_BASIC_LEVEL_PROMPT = """Generate educational assessment questions based ONLY on the following context:

CONTEXT:
{context}

INSTRUCTIONS:
1. Generate EXACTLY 2 questions total:
   - ONE question at the "remember" level of Bloom's Taxonomy
   - ONE question at the "understand" level of Bloom's Taxonomy
2. Do NOT generate questions for any other Bloom's Taxonomy levels
3. Each question must directly relate to the content in the provided context
4. Include a relevant hint and model answer for each question

BLOOM'S TAXONOMY LEVEL DEFINITIONS:
- remember: Tests recall of specific facts, terminology, or basic concepts from the text
- understand: Tests comprehension and the ability to explain ideas or concepts in one's own words

FORMAT YOUR RESPONSE AS A JSON OBJECT WITH THE EXACT STRUCTURE SHOWN IN THE SCHEMA.
"""

BLOOM_INTERMEDIATE_LEVEL_PROMPT = """Generate educational assessment questions based ONLY on the following context:

CONTEXT:
{context}

INSTRUCTIONS:
1. Generate EXACTLY 2 questions total:
   - ONE question at the "apply" level of Bloom's Taxonomy
   - ONE question at the "analyze" level of Bloom's Taxonomy
2. Do NOT generate questions for any other Bloom's Taxonomy levels (no remember, understand, evaluate or create questions)
3. Each question must directly relate to the content in the provided context
4. Include a relevant hint and model answer for each question

BLOOM'S TAXONOMY LEVEL DEFINITIONS:
- apply: Tests the ability to use learned information in new situations or to solve problems
- analyze: Tests the ability to break information into parts and examine relationships

FORMAT YOUR RESPONSE AS A JSON OBJECT WITH THE EXACT STRUCTURE SHOWN IN THE SCHEMA.
"""

BLOOM_ADVANCED_LEVEL_PROMPT = """Generate educational assessment questions based ONLY on the following context:

CONTEXT:
{context}

INSTRUCTIONS:
1. Generate EXACTLY 2 questions total:
   - ONE question at the "evaluate" level of Bloom's Taxonomy
   - ONE question at the "create" level of Bloom's Taxonomy
2. Do NOT generate questions for any other Bloom's Taxonomy levels (no remember, understand, apply, or analyze questions)
3. Each question must directly relate to the content in the provided context
4. Include a relevant hint and model answer for each question

BLOOM'S TAXONOMY LEVEL DEFINITIONS:
- evaluate: Tests the ability to make judgments, assess validity, or justify a position based on criteria
- create: Tests the ability to design new solutions, generate ideas, or synthesize information in novel ways

FORMAT YOUR RESPONSE AS A JSON OBJECT WITH THE EXACT STRUCTURE SHOWN IN THE SCHEMA.
"""

BLOOM_QUESTION_GENERATION_BASIC_SCHEMA = {
    "questions": [
        {
            "bloom_level": "remember",
            "question": "What is the key concept described in the text?",
            "hint": "Look for definitions or fundamental concepts",
            "answer": "This is a sample answer that would accurately describe the key concept from the provided context.",
        },
        {
            "bloom_level": "understand",
            "question": "Explain how the described system works.",
            "hint": "Consider the processes or mechanisms explained in the context",
            "answer": "This is a sample answer that would correctly explain the system or process based on information in the provided context.",
        }
    ]
}

BLOOM_QUESTION_GENERATION_INTERMEDIATE_SCHEMA = {
    "questions": [
        {
            "bloom_level": "apply",
            "question": "How would you use this concept to solve a specific problem?",
            "hint": "Think about practical applications of the principles discussed",
            "answer": "This is a sample answer showing how to apply the concept to solve a specific problem.",
        },
        {
            "bloom_level": "analyze",
            "question": "What are the components or relationships described in the text?",
            "hint": "Break down the concept into its constituent parts",
            "answer": "This is a sample answer analyzing the components or relationships from the provided context.",
        }
    ]
}

BLOOM_QUESTION_GENERATION_ADVANCED_SCHEMA = {
    "questions": [
        {
            "bloom_level": "evaluate",
            "question": "How effective is the approach described in the text?",
            "hint": "Consider the strengths and limitations of the concept",
            "answer": "This is a sample answer evaluating the effectiveness of the approach described in the context.",
        },
        {
            "bloom_level": "create",
            "question": "How could you design an alternative approach to address the same problem?",
            "hint": "Think about novel applications or extensions of the concept",
            "answer": "This is a sample answer proposing a creative alternative approach based on the context."
        }
    ]
}

class BloomLevel(Enum):
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class BloomAbstractLevel(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"



@dataclass
class Question:
    question: str
    bloom_level: BloomLevel
    context: Optional[str] = None
    hint: Optional[str] = None
    answer: Optional[str] = None

    @staticmethod
    def from_dict(data: Dict) -> "Question":
        return Question(
            question=data["question"],
            bloom_level=BloomLevel(data["bloom_level"]),
            hint=data.get("hint"),
            answer=data.get("answer"),
        )

    def as_dict(self) -> Dict:
        return {
            "question": self.question,
            "bloom_level": self.bloom_level.value,
            "hint": self.hint,
            "answer": self.answer,
            "context": self.context,
        }

    @staticmethod
    def from_raw_response(response: Dict, context: str) -> List["Question"]:
        """Convert a raw LLM response into a list of Question objects.
        
        Args:
            response: Raw dictionary response from LLM containing questions
            
        Returns:
            List of Question objects
        """
        questions = []
        if not isinstance(response, Dict) or "questions" not in response:
            return questions
            
        for raw in response["questions"]:
            try:
                question = Question.from_dict(raw)
                question.context = context
                questions.append(question)
            except (KeyError, ValueError) as e:
                logger.error(f"Error processing question: {e}")
                continue
                
        return questions


class Gemma3QuestionGenerator:
    def __init__(self, llm: Optional[Gemma3Model] = None):
        if llm is None:
            self.llm = Gemma3Model(
                model_name="google/gemma-3-1b-it",
                device="mps",
                torch_dtype=torch.float16,
                temperature=0.1,  # Very low temperature for predictable JSON
            )
        else:
            self.llm = llm
        self.prompt_template = None
            
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.cleanup()
        
    def cleanup(self):
        """Explicitly clean up resources to prevent memory and semaphore leaks."""
        try:
            if hasattr(self, 'llm') and self.llm is not None:
                # If we own the model (created it ourselves), clean it up
                self.llm.cleanup()
                self.llm = None
                
            logger.info("Cleaned up resources for Gemma3QuestionGenerator")
        except Exception as e:
            logger.error(f"Error during Gemma3QuestionGenerator cleanup: {e}")

    async def generate_question(
        self,
        summary_context: str,
        combined_context: str,
        bloom_level: Optional[BloomAbstractLevel] = None,
    ) -> List[Question]:
        if bloom_level is None:
            bloom_level = BloomAbstractLevel.BASIC
        
        prompt_template = self._get_prompt_template(bloom_level)
        prompt = prompt_template.format(context=combined_context)
        logger.info(f"Gemma3QuestionGenerator Prompt: {prompt}")
        
        # Select the appropriate schema based on the bloom level
        schema = self._get_schema_for_level(bloom_level)
        logger.info(f"Using schema for level: {bloom_level.value}")
        
        response = self.llm.generate(prompt, schema=schema)
        logger.info(f"Gemma3QuestionGenerator Response: {response}")
        
        # Type assertion to ensure response is a dictionary
        if not isinstance(response, Dict):
            logger.error(f"Expected dictionary response, got {type(response)}")
            return []

        return Question.from_raw_response(response, summary_context)
    
    def _get_prompt_template(self, bloom_level: BloomAbstractLevel) -> str:
        """Get the appropriate prompt template for the given BloomAbstractLevel."""
        if bloom_level == BloomAbstractLevel.BASIC:
            return BLOOM_BASIC_LEVEL_PROMPT
        elif bloom_level == BloomAbstractLevel.INTERMEDIATE:
            return BLOOM_INTERMEDIATE_LEVEL_PROMPT
        elif bloom_level == BloomAbstractLevel.ADVANCED:
            return BLOOM_ADVANCED_LEVEL_PROMPT
        else:
            return BLOOM_BASIC_LEVEL_PROMPT
        
    def _get_schema_for_level(self, bloom_level: BloomAbstractLevel) -> Dict:
        """Get the appropriate schema for the given BloomAbstractLevel."""
        if bloom_level == BloomAbstractLevel.BASIC:
            return BLOOM_QUESTION_GENERATION_BASIC_SCHEMA
        elif bloom_level == BloomAbstractLevel.INTERMEDIATE:
            return BLOOM_QUESTION_GENERATION_INTERMEDIATE_SCHEMA
        elif bloom_level == BloomAbstractLevel.ADVANCED:
            return BLOOM_QUESTION_GENERATION_ADVANCED_SCHEMA
        else:
            return BLOOM_QUESTION_GENERATION_BASIC_SCHEMA


class ExamGenerator:
    def __init__(self, llm: Union[Gemma3QuestionGenerator, Any]):
        self.llm = llm

    async def generate_exam(
        self,
        summary_context: str,
        chunks: List[str],
        bloom_level: Optional[BloomAbstractLevel] = None,
    ) -> List[Question]:
        """Generate an exam with specified number of questions per Bloom's level."""
        combined_context = "\n".join(chunks)
        logger.info("Generating exam.....")
        logger.info(f"Combined context: {combined_context}")
        logger.info(f"Summary context: {summary_context}")
        questions = await self.llm.generate_question(
            summary_context, combined_context, bloom_level
        )
        logger.info(f"Questions: {questions}")
        return questions
