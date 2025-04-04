from typing import List, Dict, Optional
from utils import logger
from core.pre_trained_model import GoogleGeminiAPI, Gemma3Model
from typing import Union, Any
from models import MultipleChoiceQuestion, BloomLevel
from models.topic_mc_quiz import TopicMcQuiz
from config import settings

MC_QUESTION_REMEMBER_PROMPT = """
Generate multiple choice questions about the following topic and context:

TOPIC: {topic}
CONTEXT: {context}

Create EXACTLY 4 multiple choice questions questions for "remember" level based on Bloom's Taxonomy.

Guidelines for "remember" level questions:
- remember: Test recall of specific facts or basic concepts

For each question:
1. Create exactly 4 answer options (A, B, C, D)
2. Mark the correct answer clearly
3. Ensure wrong options (distractors) are plausible but clearly incorrect
4. Make distractors relate to common misconceptions when possible
"""

MC_QUESTION_UNDERSTAND_PROMPT = """
Generate multiple choice questions about the following topic and context:

TOPIC: {topic}
CONTEXT: {context}

Create EXACTLY 4 multiple choice questions questions for "understand" level based on Bloom's Taxonomy.

Guidelines for "understand" level questions:
- understand: Test comprehension and the ability to explain ideas or concepts in one's own words

For each question:
1. Create exactly 4 answer options (A, B, C, D)
2. Mark the correct answer clearly
3. Ensure wrong options (distractors) are plausible but clearly incorrect
4. Make distractors relate to common misconceptions when possible
"""

MC_QUESTION_APPLY_PROMPT = """
Generate multiple choice questions about the following topic and context:

TOPIC: {topic}
CONTEXT: {context}

Create EXACTLY 4 multiple choice questions questions for "apply" level based on Bloom's Taxonomy.

Guidelines for "apply" level questions:
- apply: Test the ability to use learned information in new situations or to solve problems

For each question:
1. Create exactly 4 answer options (A, B, C, D)
2. Mark the correct answer clearly
3. Ensure wrong options (distractors) are plausible but clearly incorrect
4. Make distractors relate to common misconceptions when possible
"""

MC_QUESTION_ANALYZE_PROMPT = """
Generate multiple choice questions about the following topic and context:

TOPIC: {topic}
CONTEXT: {context}

Create EXACTLY 4 multiple choice questions questions for "analyze" level based on Bloom's Taxonomy.

Guidelines for "analyze" level questions:
- analyze: Test the ability to break information into parts and examine relationships

For each question:
1. Create exactly 4 answer options (A, B, C, D)
2. Mark the correct answer clearly
3. Ensure wrong options (distractors) are plausible but clearly incorrect
4. Make distractors relate to common misconceptions when possible
"""

MC_QUESTION_EVALUATE_PROMPT = """
Generate multiple choice questions about the following topic and context:

TOPIC: {topic}
CONTEXT: {context}

Create EXACTLY 2 multiple choice questions questions for "evaluate" level based on Bloom's Taxonomy.

Guidelines for "evaluate" level questions:
- evaluate: Test the ability to make judgments, assess validity, or justify a position based on criteria

For each question:
1. Create exactly 4 answer options (A, B, C, D)
2. Mark the correct answer clearly
3. Ensure wrong options (distractors) are plausible but clearly incorrect
4. Make distractors relate to common misconceptions when possible
"""

MC_QUESTION_CREATE_PROMPT = """
Generate multiple choice questions about the following topic and context:

TOPIC: {topic}
CONTEXT: {context}

Create EXACTLY 2 multiple choice questions questions for "create" level based on Bloom's Taxonomy.

Guidelines for "create" level questions:
- create: Test the ability to design new solutions, generate ideas, or synthesize information in novel ways

For each question:
1. Create exactly 4 answer options (A, B, C, D)
2. Mark the correct answer clearly
"""

MC_QUESTION_REMEMBER_SCHEMA = {
    "questions": [
        {
            "bloom_level": "remember",
            "question": "What is the key concept described in the text?",
            "options": {
                "A": "This is a option A",
                "B": "This is a option B",
                "C": "This is a option C",
                "D": "This is a option D",
            },
            "answer": "A",
            "hint": "Look for definitios or fundamental concepts",
            "explanation": "This is a sample answer that would accurately explain why this is the correct answer.",
        }
    ]
}

MC_QUESTION_UNDERSTAND_SCHEMA = {
    "questions": [
        {
            "bloom_level": "understand",
            "question": "Explain how the described system works.",
            "options": {
                "A": "This is a option A",
                "B": "This is a option B",
                "C": "This is a option C",
                "D": "This is a option D",
            },
            "answer": "A",
            "hint": "Consider the processes or mechanisms explained in the context",
            "explanation": "This is a sample answer that would correctly explain the system or process based on information in the provided context.",
        }
    ]
}

MC_QUESTION_APPLY_SCHEMA = {
    "questions": [
        {
            "bloom_level": "apply",
            "question": "How would you use this concept to solve a specific problem?",
            "options": {
                "A": "This is a option A",
                "B": "This is a option B",
                "C": "This is a option C",
                "D": "This is a option D",
            },
            "answer": "A",
            "hint": "Think about practical applications of the principles discussed",
            "explanation": "This is a sample answer showing how to apply the concept to solve a specific problem.",
        }
    ]
}

MC_QUESTION_ANALYZE_SCHEMA = {
    "questions": [
        {
            "bloom_level": "analyze",
            "question": "What are the components or relationships described in the text?",
            "options": {
                "A": "This is a option A",
                "B": "This is a option B",
                "C": "This is a option C",
                "D": "This is a option D",
            },
            "answer": "A",
            "hint": "Break down the concept into its constituent parts",
            "explanation": "This is a sample answer analyzing the components or relationships from the provided context.",
        }
    ]
}

MC_QUESTION_EVALUATE_SCHEMA = {
    "questions": [
        {
            "bloom_level": "evaluate",
            "question": "How effective is the approach described in the text?",
            "options": {
                "A": "This is a option A",
                "B": "This is a option B",
                "C": "This is a option C",
                "D": "This is a option D",
            },
            "answer": "A",
            "hint": "Consider the strengths and limitations of the concept",
            "explanation": "This is a sample answer evaluating the effectiveness of the approach described in the context.",
        }
    ]
}

MC_QUESTION_CREATE_SCHEMA = {
    "questions": [
        {
            "bloom_level": "create",
            "question": "How could you design an alternative approach to address the same problem?",
            "options": {
                "A": "This is a option A",
                "B": "This is a option B",
                "C": "This is a option C",
                "D": "This is a option D",
            },
            "answer": "A",
            "hint": "Think about novel applications or extensions of the concept",
            "explanation": "This is a sample answer proposing a creative alternative approach based on the context.",
        }
    ]
}


class GoogleGeminiMCGenerator:
    def __init__(self, llm: Optional[Gemma3Model] = None):
        if llm is None:
            self.llm = GoogleGeminiAPI(
                api_client=settings.GOOGLE_GEMINI_CLIENT,
                temperature=0.1,
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
            if hasattr(self, "llm") and self.llm is not None:
                # If we own the model (created it ourselves), clean it up
                self.llm.cleanup()

            logger.info("Cleaned up resources for GoogleGeminiMCGenerator")
        except Exception as e:
            logger.error(f"Error during GoogleGeminiMCGenerator cleanup: {e}")

    async def generate_mc_questions(
        self,
        topic: str,
        context: str,
        bloom_level: Optional[BloomLevel] = None,
        temperature: float = 0.3,
    ) -> List[MultipleChoiceQuestion]:
        if bloom_level is None:
            bloom_level = BloomLevel.REMEMBER

        prompt_template = self._get_prompt_template(bloom_level)
        prompt = prompt_template.format(topic=topic, context=context)
        logger.info(f"GoogleGeminiMCGenerator Prompt: {prompt}")

        # Select the appropriate schema based on the bloom level
        schema = self._get_schema(bloom_level)
        logger.info(f"Using schema for level: {bloom_level.value}")

        response = self.llm.generate(prompt, schema=schema, temperature=temperature)
        logger.info(f"GoogleGeminiMCGenerator Response: {response}")

        # Type assertion to ensure response is a dictionary
        if not isinstance(response, Dict):
            logger.error(f"Expected dictionary response, got {type(response)}")
            return []

        return MultipleChoiceQuestion.from_raw_response_list(response)

    def _get_prompt_template(self, bloom_level: BloomLevel) -> str:
        """Get the appropriate prompt template for the given BloomAbstractLevel."""
        if bloom_level == BloomLevel.REMEMBER:
            return MC_QUESTION_REMEMBER_PROMPT
        elif bloom_level == BloomLevel.UNDERSTAND:
            return MC_QUESTION_UNDERSTAND_PROMPT
        elif bloom_level == BloomLevel.APPLY:
            return MC_QUESTION_APPLY_PROMPT
        elif bloom_level == BloomLevel.ANALYZE:
            return MC_QUESTION_ANALYZE_PROMPT
        elif bloom_level == BloomLevel.EVALUATE:
            return MC_QUESTION_EVALUATE_PROMPT
        elif bloom_level == BloomLevel.CREATE:
            return MC_QUESTION_CREATE_PROMPT
        else:
            raise ValueError(f"Invalid bloom level: {bloom_level}")

    def _get_schema(self, bloom_level: BloomLevel) -> Dict:
        """Get the appropriate schema for the given BloomLevel."""
        if bloom_level == BloomLevel.REMEMBER:
            return MC_QUESTION_REMEMBER_SCHEMA
        elif bloom_level == BloomLevel.UNDERSTAND:
            return MC_QUESTION_UNDERSTAND_SCHEMA
        elif bloom_level == BloomLevel.APPLY:
            return MC_QUESTION_APPLY_SCHEMA
        elif bloom_level == BloomLevel.ANALYZE:
            return MC_QUESTION_ANALYZE_SCHEMA
        elif bloom_level == BloomLevel.EVALUATE:
            return MC_QUESTION_EVALUATE_SCHEMA
        elif bloom_level == BloomLevel.CREATE:
            return MC_QUESTION_CREATE_SCHEMA


class EssayGenerator:
    def __init__(self, llm: Union[GoogleGeminiMCGenerator, Any]):
        self.llm = llm

    async def generate_mc_questions(
        self,
        topic: str,
        context: str,
        bloom_level: Optional[BloomLevel] = None,
        temperature: float = 0.3,
    ) -> List[MultipleChoiceQuestion]:
        """Generate an exam with specified number of questions per Bloom's level."""
        # Use existing chunks for context
        questions = await self.llm.generate_mc_questions(
            topic, context, bloom_level, temperature
        )
        logger.info(f"Questions: {questions}")
        return questions
