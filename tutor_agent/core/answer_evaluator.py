from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json

class AnswerEvaluator:
    """Evaluates student answers to exam questions."""

    ANSWER_EVALUATION_PROMPT = """You are evaluating a student's answer to an exam question.

Question: {question}
Expected Key Points: {key_points}
Student's Answer: {student_answer}

Evaluate the answer and provide:
1. Score (0-100)
2. Detailed feedback
3. Missing key points
4. Suggestions for improvement

Format your response as a JSON object:
{
  "score": 85,
  "feedback": "Your detailed feedback here",
  "missing_points": ["missing point 1", "missing point 2"],
  "suggestions": ["suggestion 1", "suggestion 2"]
}"""

    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model_name)
        self.evaluation_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["question", "key_points", "student_answer"],
                template=self.ANSWER_EVALUATION_PROMPT
            )
        )
    
    async def evaluate_answer(self, question: str, key_points: list, student_answer: str) -> Dict:
        """Evaluate a student's answer to a question."""
        result = await self.evaluation_chain.ainvoke({
            "question": question,
            "key_points": json.dumps(key_points) if isinstance(key_points, list) else key_points,
            "student_answer": student_answer
        })
        
        try:
            # Parse the JSON response
            evaluation = json.loads(result["text"])
            return evaluation
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "score": 0,
                "feedback": "Unable to evaluate answer. Please try again.",
                "missing_points": [],
                "suggestions": ["Provide a more detailed answer"]
            } 