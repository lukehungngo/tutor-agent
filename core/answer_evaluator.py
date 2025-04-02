from typing import List, Dict, Optional, Union, Any
from utils import logger, time_execution
from models import CorrectnessLevel, EvaluationResult
from core.pre_trained_model.google_gemma import Gemma3Model
import torch

EVALUATION_PROMPT = """Evaluate the the answer as a supportive tutor:

QUESTION CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

As a supportive tutor, provide learning-focused feedback that:
1. Clearly identifies if the answer is correct, or partially correct, or incorrect
2. Lists specific accurate parts of the answer
3. Offers concrete suggestions for improvement
4. Provides encouraging feedback that acknowledges effort
5. Suggests next steps for continued learning

FORMAT YOUR RESPONSE AS A JSON OBJECT WITH THE EXACT STRUCTURE SHOWN IN THE SCHEMA.
"""

EVALUATION_SCHEMA = {
    "correctness_level": ["correct", "partially_correct", "incorrect"],
    "accurate_parts": ["Accurate part example"],
    "improvement_suggestions": ["Suggestion example"],
    "encouragement": "Encouragement example",
    "next_steps": "Next steps example",
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
            self.llm = Gemma3Model(
                model_name="google/gemma-3-1b-it",
                device="mps",
                torch_dtype=torch.float16,
                temperature=0.2,  # Very low temperature for predictable JSON
            )
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
            response = self.llm.generate(prompt, schema=EVALUATION_SCHEMA)
            logger.info(f"Gemma3AnswerEvaluator Response: {response}")

            # Type assertion to ensure response is a dictionary
            if not isinstance(response, Dict):
                logger.error(f"Expected dictionary response, got {type(response)}")
                return self._default_error_evaluation()

            # Create an EvaluationResult from the response
            return EvaluationResult.from_raw_response(response)

        except Exception as e:
            logger.error(f"Error during answer evaluation: {e}")
            return self._default_error_evaluation()

    def _default_error_evaluation(self) -> EvaluationResult:
        """
        Create a default evaluation result for error situations.

        Returns:
            A default EvaluationResult object
        """
        return EvaluationResult(
            correctness_level=CorrectnessLevel.PARTIALLY_CORRECT,
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
        self, context: str, question: str, student_answer: str
    ) -> EvaluationResult:
        """
        Evaluate a student's answer using the configured evaluator.

        Args:
            question: The question that was asked
            student_answer: The student's submitted answer

        Returns:
            An EvaluationResult object containing the evaluation details
        """
        return await self.llm.evaluate_answer(context, question, student_answer)
