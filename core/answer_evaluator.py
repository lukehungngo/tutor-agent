from typing import List, Dict, Optional, Union, Any
from utils import logger, time_execution
from models import CorrectnessLevel, EvaluationResult, BloomAbstractLevel
from core.pre_trained_model.google_gemma import Gemma3Model
from core.pre_trained_model.google_gemini_api import GoogleGeminiAPI

EVALUATION_PROMPT = """
Evaluate the student's answer using both the provided context and your general knowledge of the subject.

QUESTION CONTEXT: {context}
QUESTION: {question}
STUDENT ANSWER: {answer}

EVALUATION INSTRUCTIONS:
1. First assess answer evaluability:
   - If irrelevant/nonsensical: mark "incorrect" with score=0

2. For meaningful answers, evaluate using:
   - Context-specific accuracy: Does it match information from the provided context?
   - General subject knowledge: Is it accurate according to broader domain knowledge?
   - Completeness: Are all key elements addressed?

   Correctness:
   - "correct": Accurate per context AND general knowledge, with all key elements
   - "partially_correct": Some accuracy but with gaps or minor errors
   - "incorrect": Major errors or misunderstandings

   Quality Score (0-100):
   - 90-100: Exceptional - comprehensive, precise, shows deep understanding
   - 70-89: Good - accurate but missing some details or nuance
   - 50-69: Basic - contains core concepts but lacks depth or precision
   - 20-49: Limited - has some relevant elements but significant gaps
   - 1-19: Minimal - barely addresses the question
   - 0: Irrelevant or nonsensical

   Constraints:
   - Brief answers: Maximum score of 70
   - Scores above 90 require exceptional clarity and comprehensive coverage

3. Score-correctness alignment:
   - Score 0: Must be "incorrect"
   - Scores 1-49: Must be "incorrect" or "partially_correct"
   - Scores 50-79: Can be any correctness level based on accuracy
   - Scores 80-100: Must be "partially_correct" or "correct"

4. IMPORTANT VALIDATION:
   - If feedback mentions correct elements, correctness_level CANNOT be "incorrect" with score 0
   - If score is 0, next_steps and encouragement should indicate complete rework needed
   - Always check consistency between your assessment and the final scoring

FORMAT YOUR RESPONSE AS A JSON OBJECT WITH THE EXACT STRUCTURE SHOWN IN THE SCHEMA.
"""


EVALUATION_SCHEMA = {
    "correctness_level": "correct|partially_correct|incorrect",
    "score": 0,
    "accurate_parts": [
        "Identify specific correct elements in the answer based on the context (if any)"
    ],
    "improvement_suggestions": ["Suggest how to improve the answer"],
    "encouragement": "Provide encouraging feedback appropriate to the quality of the answer",
    "next_steps": "Suggest specific next steps for learning",
}


class Gemma3AnswerEvaluator:
    """Evaluates student answers with learning-focused feedback using Gemma3 model."""

    def __init__(self, llm: Optional[Gemma3Model] = None):
        """
        Initialize the evaluator with a Gemma3 language model.

        Args:
            llm: The language model to use for evaluation, or None to create a new one
        """
        if llm is None:
            self.llm = GoogleGeminiAPI()
        else:
            self.llm = llm

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.cleanup()

    def cleanup(self):
        """Explicitly clean up resources to prevent memory and semaphore leaks."""
        try:
            if hasattr(self, "llm") and self.llm is not None:
                # If we own the model (created it ourselves), clean it up
                self.llm.cleanup()

            logger.info("Cleaned up resources for Gemma3AnswerEvaluator")
        except Exception as e:
            logger.error(f"Error during Gemma3AnswerEvaluator cleanup: {e}")

    @time_execution
    async def evaluate_answer(
        self,
        context: str,
        question: str,
        answer: str,
        temperature: float = 0.2,
    ) -> EvaluationResult:
        """
        Evaluate a student's answer with learning-focused feedback.

        Args:
            question: The question that was asked
            model_answer: The expected correct answer
            student_answer: The student's submitted answer

        Returns:
            An EvaluationResult object containing the evaluation details
        """
        try:
            # Format the prompt with the question and answers
            prompt = EVALUATION_PROMPT.format(
                context=context,
                question=question,
                answer=answer,
            )

            logger.info(f"Gemma3AnswerEvaluator Prompt: {prompt}")

            # Generate evaluation using the language model
            response = self.llm.generate(
                prompt, schema=EVALUATION_SCHEMA, temperature=temperature
            )
            print(f"Gemma3AnswerEvaluator Response: {response}")
            # Type assertion to ensure response is a dictionary
            if not isinstance(response, Dict):
                logger.error(f"Expected dictionary response, got {type(response)}")
                return self._default_error_evaluation()

            # Create an EvaluationResult from the response
            return EvaluationResult.from_raw_response(response)

        except Exception as e:
            logger.error(f"Error during answer evaluation: {e}")
            raise e

    def _default_error_evaluation(self) -> EvaluationResult:
        """
        Create a default evaluation result for error situations.

        Returns:
            A default EvaluationResult object
        """
        return EvaluationResult(
            correctness_level=CorrectnessLevel.PARTIALLY_CORRECT,
            score=0,
            accurate_parts=[],
            improvement_suggestions=["Please try submitting your answer again"],
            encouragement="Thank you for your patience.",
            next_steps="You might want to review the material while we resolve this issue.",
        )


class AnswerEvaluator:
    """Main class for evaluating student answers."""

    def __init__(self, llm: Union[Gemma3AnswerEvaluator, Any]):
        """
        Initialize the answer evaluator.

        Args:
            llm: The language model evaluator to use
        """
        self.llm = llm

    async def evaluate_answer(
        self, context: str, question: str, student_answer: str, temperature: float = 0.2
    ) -> EvaluationResult:
        """
        Evaluate a student's answer using the configured evaluator.

        Args:
            question: The question that was asked
            student_answer: The student's submitted answer

        Returns:
            An EvaluationResult object containing the evaluation details
        """
        return await self.llm.evaluate_answer(context, question, student_answer, temperature=temperature)
