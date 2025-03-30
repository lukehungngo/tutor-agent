from typing import List, Dict, Any
from enum import Enum
from dataclasses import dataclass
from utils import logger


class CorrectnessLevel(Enum):
    """Enum representing the correctness level of an answer."""

    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"


@dataclass
class EvaluationResult:
    """Data class for storing the result of an answer evaluation."""

    correctness_level: CorrectnessLevel
    accurate_parts: List[str]
    improvement_suggestions: List[str]
    encouragement: str
    next_steps: str

    @staticmethod
    def from_dict(data: Dict) -> "EvaluationResult":
        """
        Create an EvaluationResult from a dictionary.

        Args:
            data: Dictionary containing evaluation data

        Returns:
            EvaluationResult object
        """
        try:
            correctness_level = CorrectnessLevel(
                data.get("correctness_level", "partially_correct")
            )
        except (ValueError, KeyError):
            correctness_level = CorrectnessLevel.PARTIALLY_CORRECT

        return EvaluationResult(
            correctness_level=correctness_level,
            accurate_parts=data.get("accurate_parts", []),
            improvement_suggestions=data.get("improvement_suggestions", []),
            encouragement=data.get("encouragement", "Good effort!"),
            next_steps=data.get("next_steps", "Review the related concepts."),
        )

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the evaluation result to a dictionary.

        Returns:
            Dictionary representation of the evaluation result
        """
        return {
            "correctness_level": self.correctness_level.value,
            "accurate_parts": self.accurate_parts,
            "improvement_suggestions": self.improvement_suggestions,
            "encouragement": self.encouragement,
            "next_steps": self.next_steps,
        }

    @staticmethod
    def from_raw_response(response: Dict) -> "EvaluationResult":
        """
        Convert a raw LLM response into an EvaluationResult object.

        Args:
            response: Raw dictionary response from LLM containing evaluation

        Returns:
            EvaluationResult object
        """
        try:
            return EvaluationResult.from_dict(response)
        except Exception as e:
            logger.error(f"Error processing evaluation response: {e}")
            # Return a default evaluation in case of error
            return EvaluationResult(
                correctness_level=CorrectnessLevel.PARTIALLY_CORRECT,
                accurate_parts=[],
                improvement_suggestions=["Please try again"],
                encouragement="Thank you for your submission.",
                next_steps="Please try again or contact support if the issue persists.",
            )
