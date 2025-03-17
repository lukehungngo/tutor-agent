from typing import Dict, List, Optional
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import random

class DocumentAnalyzer:
    """Analyzes documents to identify topics and sections for exam questions."""

    DOCUMENT_ANALYSIS_PROMPT = """You are an expert educator analyzing a document to create an exam.
Based on the following document samples:
{document_samples}

Your task is to:
1. Identify the main topics and subtopics covered in the document
2. For each topic, extract key concepts, facts, and principles that would be important to test
3. Suggest different sections of the document that should be covered in an exam

Format your response as a JSON object with the following structure:
{
  "main_topics": [
    {
      "topic": "Topic name",
      "key_concepts": ["concept1", "concept2", ...],
      "importance": "high/medium/low"
    },
    ...
  ],
  "document_sections": [
    {
      "section": "Section name or description",
      "query": "search query to find this section"
    },
    ...
  ]
}

Focus on providing a comprehensive overview that covers the breadth of the document."""

    CHUNK_ANALYSIS_PROMPT = """You are an expert educator analyzing a document chunk to create exam questions.
Based on the following document chunk:
{chunk_content}

Your task is to:
1. Identify the main topic of this chunk
2. Extract key concepts, facts, and principles that would be important to test
3. Determine the importance of this chunk for an exam (high/medium/low)

Format your response as a JSON object with the following structure:
{
  "topic": "Main topic of this chunk",
  "key_concepts": ["concept1", "concept2", ...],
  "importance": "high/medium/low",
  "question_potential": "Brief explanation of what kinds of questions could be generated from this chunk"
}"""

    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model_name)
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["document_samples"],
                template=self.DOCUMENT_ANALYSIS_PROMPT
            )
        )
        self.chunk_analysis_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["chunk_content"],
                template=self.CHUNK_ANALYSIS_PROMPT
            )
        )
        self.analysis_strategy = "holistic"  # Default strategy
    
    async def analyze_document(self, document_samples: List[Document], 
                              strategy: str = "holistic",
                              user_highlights: Optional[Dict] = None) -> Dict:
        """
        Analyze document samples to identify topics and sections.
        
        Args:
            document_samples: List of document chunks to analyze
            strategy: Analysis strategy - "holistic", "chunk-based", or "user-assisted"
            user_highlights: Optional dict of user-provided highlights/tags
            
        Returns:
            Document analysis as a dictionary
        """
        self.analysis_strategy = strategy
        
        if strategy == "chunk-based":
            return await self._analyze_chunk_based(document_samples)
        elif strategy == "user-assisted" and user_highlights:
            return await self._analyze_user_assisted(document_samples, user_highlights)
        else:
            # Default to holistic analysis
            return await self._analyze_holistic(document_samples)
    
    async def _analyze_holistic(self, document_samples: List[Document]) -> Dict:
        """Original holistic analysis approach."""
        # Prepare samples for analysis
        samples_text = "\n\n---\n\n".join([doc.page_content for doc in document_samples[:8]])
        
        # Analyze document structure
        response = await self.analysis_chain.ainvoke({"document_samples": samples_text})
        
        try:
            # Parse the JSON response
            analysis = json.loads(response["text"])
            return analysis
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "main_topics": [{"topic": "General Content", "key_concepts": [], "importance": "high"}],
                "document_sections": [{"section": "Main Content", "query": "main content"}]
            }
    
    async def _analyze_chunk_based(self, document_chunks: List[Document]) -> Dict:
        """
        Chunk-based analysis approach.
        Analyzes each chunk independently and then aggregates the results.
        """
        # Calculate number of chunks to analyze based on document size
        def calculate_chunks_to_analyze(document_chunks: List[Document]) -> int:
            total_chunks = len(document_chunks)
            
            # Base number - analyze at least 15 chunks
            base_chunks = 15
            
            # For larger documents, analyze more chunks but with diminishing returns
            if total_chunks > 100:
                # For very large documents, analyze 10% of chunks, but cap at 50
                return min(base_chunks + int(total_chunks * 0.1), 50)
            elif total_chunks > 50:
                # For medium documents, analyze 20% of chunks
                return min(base_chunks + int(total_chunks * 0.2), 30)
            else:
                # For small documents, analyze up to 15 chunks or all chunks
                return min(total_chunks, base_chunks)
        
        max_chunks_to_analyze = calculate_chunks_to_analyze(document_chunks)
        
        # Select a representative sample of chunks if there are too many
        chunks_to_analyze = document_chunks
        if len(document_chunks) > max_chunks_to_analyze:
            chunks_to_analyze = random.sample(document_chunks, max_chunks_to_analyze)
        
        # Analyze each chunk independently
        chunk_analyses = []
        for chunk in chunks_to_analyze:
            try:
                response = await self.chunk_analysis_chain.ainvoke({"chunk_content": chunk.page_content})
                chunk_analysis = json.loads(response["text"])
                chunk_analysis["source"] = chunk.metadata.get("source", "unknown")
                chunk_analyses.append(chunk_analysis)
            except (json.JSONDecodeError, Exception) as e:
                # Skip problematic chunks
                continue
        
        # Aggregate the results
        topics_map = {}
        for analysis in chunk_analyses:
            topic = analysis.get("topic", "")
            if not topic:
                continue
                
            if topic in topics_map:
                # Merge key concepts
                topics_map[topic]["key_concepts"].extend(analysis.get("key_concepts", []))
                # Update importance if higher
                importance_values = {"high": 3, "medium": 2, "low": 1}
                current_importance = topics_map[topic]["importance"]
                new_importance = analysis.get("importance", "low")
                if importance_values.get(new_importance, 0) > importance_values.get(current_importance, 0):
                    topics_map[topic]["importance"] = new_importance
            else:
                topics_map[topic] = {
                    "topic": topic,
                    "key_concepts": analysis.get("key_concepts", []),
                    "importance": analysis.get("importance", "medium")
                }
        
        # Remove duplicate key concepts
        for topic in topics_map:
            topics_map[topic]["key_concepts"] = list(set(topics_map[topic]["key_concepts"]))
        
        # Format the final analysis
        main_topics = list(topics_map.values())
        
        # Create document sections based on chunks
        document_sections = []
        for analysis in chunk_analyses:
            if "question_potential" in analysis:
                document_sections.append({
                    "section": analysis.get("topic", ""),
                    "query": analysis.get("topic", ""),
                    "question_potential": analysis.get("question_potential", "")
                })
        
        return {
            "main_topics": main_topics,
            "document_sections": document_sections,
            "analysis_method": "chunk-based"
        }
    
    async def _analyze_user_assisted(self, document_samples: List[Document], user_highlights: Dict) -> Dict:
        """
        User-assisted analysis approach.
        Prioritizes user-highlighted sections and tags.
        
        Expected user_highlights format:
        {
            "highlighted_sections": [
                {"text": "...", "importance": "high"},
                ...
            ],
            "tags": [
                {"tag": "Important Concept", "related_terms": ["term1", "term2"]},
                ...
            ],
            "summary": "Optional user summary of the document"
        }
        """
        # Extract user highlights
        highlighted_sections = user_highlights.get("highlighted_sections", [])
        user_tags = user_highlights.get("tags", [])
        user_summary = user_highlights.get("summary", "")
        
        # Create main topics from user tags
        main_topics = []
        for tag in user_tags:
            main_topics.append({
                "topic": tag.get("tag", ""),
                "key_concepts": tag.get("related_terms", []),
                "importance": "high",  # User-tagged items are considered high importance
                "user_tagged": True
            })
        
        # Create document sections from highlighted sections
        document_sections = []
        for section in highlighted_sections:
            document_sections.append({
                "section": section.get("text", "")[:50] + "...",  # Use beginning of text as section name
                "query": section.get("text", ""),
                "importance": section.get("importance", "high"),
                "user_highlighted": True
            })
        
        # If we have very few user inputs, supplement with holistic analysis
        if len(main_topics) < 3 and len(document_sections) < 3:
            # Get holistic analysis
            holistic_analysis = await self._analyze_holistic(document_samples)
            
            # Merge with user inputs, prioritizing user inputs
            existing_topics = {topic["topic"] for topic in main_topics}
            for topic in holistic_analysis.get("main_topics", []):
                if topic["topic"] not in existing_topics:
                    topic["user_tagged"] = False
                    main_topics.append(topic)
            
            existing_sections = {section["query"] for section in document_sections}
            for section in holistic_analysis.get("document_sections", []):
                if section["query"] not in existing_sections:
                    section["user_highlighted"] = False
                    document_sections.append(section)
        
        return {
            "main_topics": main_topics,
            "document_sections": document_sections,
            "user_summary": user_summary,
            "analysis_method": "user-assisted"
        }
    
    def get_document_samples(self, vector_store, max_samples: int = 8) -> List[Document]:
        """Get representative samples from different parts of the document."""
        # Beginning samples
        beginning_docs = vector_store.similarity_search("introduction beginning overview", k=2)
        
        # Middle samples - use a generic query that's likely to match middle content
        middle_docs = vector_store.similarity_search("main content key concepts", k=2)
        
        # End samples
        end_docs = vector_store.similarity_search("conclusion summary findings", k=2)
        
        # Random samples with diverse queries
        diverse_queries = [
            "important definition",
            "critical analysis",
            "key example",
            "methodology process",
            "results findings"
        ]
        random_docs = []
        for query in diverse_queries:
            results = vector_store.similarity_search(query, k=1)
            if results:
                random_docs.extend(results)
        
        # Combine all samples
        all_samples = beginning_docs + middle_docs + end_docs + random_docs
        
        # Remove duplicates
        unique_samples = []
        seen_content = set()
        for doc in all_samples:
            if doc.page_content not in seen_content:
                unique_samples.append(doc)
                seen_content.add(doc.page_content)
        
        return unique_samples[:max_samples] 