import asyncio
from typing import List
import os
from langchain.chat_models import ChatOpenAI
from core.exam_generator import ExamGenerator, BloomLevel


async def main():
    # Example document chunks
    chunks = [
        """The water cycle, also known as the hydrologic cycle, describes the continuous movement of water 
        within the Earth and atmosphere. It is a complex system that includes many different processes: 
        evaporation, condensation, precipitation, and runoff.""",
        """Photosynthesis is the process by which plants convert light energy into chemical energy. 
        The process takes place in the chloroplasts of plant cells and requires sunlight, carbon dioxide, 
        and water as inputs.""",
        """Climate change refers to long-term shifts in global weather patterns. The main cause is the 
        increase in greenhouse gases in Earth's atmosphere, primarily from burning fossil fuels. This leads 
        to rising global temperatures and various environmental impacts.""",
    ]

    # Configure how many questions you want per Bloom's level
    questions_per_level = {
        BloomLevel.REMEMBER: 1,  # Basic recall
        BloomLevel.UNDERSTAND: 1,  # Comprehension
        BloomLevel.APPLY: 1,  # Application
        BloomLevel.ANALYZE: 1,  # Analysis
        BloomLevel.EVALUATE: 1,  # Evaluation
        BloomLevel.CREATE: 1,  # Creation
    }

    # Initialize the LLM
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

    # Initialize the exam generator
    generator = ExamGenerator(llm=llm)

    # Generate the exam
    print("Generating exam questions...")
    questions = await generator.generate_exam(chunks, questions_per_level)

    # Print the generated questions
    print("\nGenerated Exam Questions:")
    print("------------------------\n")

    for i, question in enumerate(questions, 1):
        print(f"Question {i} ({question.bloom_level.value.capitalize()} Level):")
        print(f"Q: {question.question}")
        print(f"Suggested Answer: {question.suggested_answer}")
        print("------------------------\n")


if __name__ == "__main__":
    asyncio.run(main())
