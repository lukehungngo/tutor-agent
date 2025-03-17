from typing import List, Dict, Optional, Any
from langchain.schema import Document
from .vector_store import VectorStoreManager
from .document_processor import DocumentProcessor
from .document_analyzer import DocumentAnalyzer
from .question_generator import QuestionGenerator
from .answer_evaluator import AnswerEvaluator

class ExamSession:
    """Manages an exam session with document-based questions."""

    def __init__(self, model_name: str = "gpt-4"):
        self.vector_store = VectorStoreManager()
        self.doc_processor = DocumentProcessor()
        self.document_analyzer = DocumentAnalyzer(model_name=model_name)
        self.question_generator = QuestionGenerator(model_name=model_name)
        self.answer_evaluator = AnswerEvaluator(model_name=model_name)
        self.current_questions: List[Dict] = []
        self.document_analysis: Dict = {}
        self.analysis_strategy = "holistic"  # Default strategy
        
    async def initialize_exam(self, document_path: str, 
                             summary: Optional[str] = None, 
                             num_questions: int = 5,
                             analysis_strategy: str = "holistic",
                             user_highlights: Optional[Dict] = None) -> List[Dict]:
        """Initialize an exam session from a document with optional summary."""
        # Set analysis strategy
        self.analysis_strategy = analysis_strategy
        
        # Process and store document
        documents = self.doc_processor.load_document(document_path)
        
        # Process with summary if provided
        if summary:
            chunks = self.doc_processor.process_with_summary(documents, summary)
        else:
            chunks = self.doc_processor.split_documents(documents)
            
        self.vector_store.create_vector_store(chunks)
        
        # Create user highlights dict if we have a summary but no explicit highlights
        if summary and not user_highlights:
            user_highlights = {
                "highlighted_sections": [],
                "tags": [],
                "summary": summary
            }
        
        # Analyze document to identify topics and sections
        await self._analyze_document(strategy=analysis_strategy, user_highlights=user_highlights)
        
        # Generate initial set of questions
        questions = await self.generate_exam_questions(num_questions)
        return questions
    
    async def _analyze_document(self, strategy: str = "holistic", user_highlights: Optional[Dict] = None) -> Dict:
        """Analyze the document to identify topics and sections for comprehensive coverage."""
        # Get document samples based on strategy
        if strategy == "chunk-based":
            # For chunk-based, we need more chunks
            all_chunks = self.vector_store.get_all_documents()
            analysis = await self.document_analyzer.analyze_document(
                all_chunks, 
                strategy="chunk-based"
            )
        elif strategy == "user-assisted" and user_highlights:
            # For user-assisted, we use the user highlights
            document_samples = self.document_analyzer.get_document_samples(self.vector_store)
            analysis = await self.document_analyzer.analyze_document(
                document_samples,
                strategy="user-assisted",
                user_highlights=user_highlights
            )
        else:
            # Default holistic approach
            document_samples = self.document_analyzer.get_document_samples(self.vector_store)
            analysis = await self.document_analyzer.analyze_document(document_samples)
        
        self.document_analysis = analysis
        return analysis
    
    async def generate_exam_questions(self, num_questions: int) -> List[Dict]:
        """Generate a set of exam questions with good coverage of the document."""
        # If we have document analysis, use it to generate questions with good coverage
        if not self.document_analysis:
            await self._analyze_document()
        
        # Get topics from document analysis
        topics = self.document_analysis.get("main_topics", [])
        if not topics:
            topics = [{"topic": "General Content", "importance": "high"}]
        
        # Generate questions for topics
        questions = await self.question_generator.generate_questions_for_topics(
            self.vector_store, 
            topics, 
            num_questions
        )
        
        self.current_questions = questions
        return questions
    
    async def evaluate_answer(self, question_idx: int, student_answer: str) -> Dict:
        """Evaluate a student's answer to a specific question."""
        if not (0 <= question_idx < len(self.current_questions)):
            raise ValueError("Invalid question index")
        
        question_data = self.current_questions[question_idx]
        
        evaluation = await self.answer_evaluator.evaluate_answer(
            question=question_data.get("question", ""),
            key_points=question_data.get("key_points", []),
            student_answer=student_answer
        )
        
        return evaluation 