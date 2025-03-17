from typing import List, Dict, Any
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json


class QuestionGenerator:
    """Generates questions based on Bloom's Taxonomy levels."""

    QUESTION_GENERATION_PROMPT = """Based on the following context:
{context}

Generate an exam question that tests the student's knowledge at the {level} level of Bloom's Taxonomy.
The question should be:
1. Specific to the given context
2. Clear and unambiguous
3. Appropriate for the cognitive level
4. Related to the topic: {topic}

Include:
1. Question
2. Expected answer key points
3. Cognitive level being tested

Format the response as a JSON object:
{
  "question": "The question text",
  "level": "The Bloom's taxonomy level",
  "key_points": ["key point 1", "key point 2", ...],
  "topic": "The main topic this question covers"
}"""

    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model_name)
        self.question_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["context", "level", "topic"],
                template=self.QUESTION_GENERATION_PROMPT
            )
        )
        self.bloom_levels = ["remember", "understand", "apply", "analyze", "evaluate", "create"]

    async def generate_question(self, context: str, level: str, topic: str) -> Dict:
        """Generate a question for a specific Bloom's level and topic."""
        if level not in self.bloom_levels:
            raise ValueError(f"Invalid Bloom's level. Choose from: {self.bloom_levels}")
        
        response = await self.question_chain.ainvoke({
            "context": context,
            "level": level,
            "topic": topic
        })
        
        try:
            # Parse the JSON response
            question_data = json.loads(response["text"])
            return question_data
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "question": response["text"],
                "level": level,
                "key_points": ["Key information from the document"],
                "topic": topic
            }

    async def generate_questions_for_topics(self, vector_store, topics: List[Dict], num_questions: int) -> List[Dict]:
        """Generate questions for a list of topics."""
        questions = []
        
        # Weight topics by importance
        weighted_topics = []
        for topic in topics:
            weight = {"high": 3, "medium": 2, "low": 1}.get(topic.get("importance", "medium"), 2)
            for _ in range(weight):
                weighted_topics.append(topic)
        
        # Select topics for questions (with replacement for important topics)
        import random
        selected_topics = random.choices(weighted_topics, k=num_questions) if weighted_topics else [{"topic": "General Content"}]
        
        # Generate questions for each selected topic
        for i, topic_data in enumerate(selected_topics):
            topic = topic_data.get("topic", "General Content")
            
            # Get relevant chunks for this specific topic
            relevant_docs = vector_store.similarity_search(topic, k=2)
            context = "\n".join(doc.page_content for doc in relevant_docs)
            
            # Select Bloom's level for this question
            level = self.bloom_levels[i % len(self.bloom_levels)]
            
            # Generate question
            question_data = await self.generate_question(context, level, topic)
            questions.append(question_data)
        
        return questions
