import os
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredFileLoader,
    CSVLoader,
)
from langchain.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from pre_trained_model import phi35mini, GenerationConfig
from utils import time_execution, logger
import torch

class DocumentProcessor:
    """A simple document processor that handles loading, chunking, and vector storage."""

    # 3000 characters each to one page of the document on average
    def __init__(self, chunk_size: int = 3000, chunk_overlap: int = 500):
        """
        Initialize the document processor.

        Args:
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.embeddings : Embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "mps"},  # Remove torch_dtype parameter
            encode_kwargs={
                "batch_size": 32,  # Reduced from 128 for better stability
                "normalize_embeddings": True,
                "convert_to_tensor": True,  # Set to False if you want numpy arrays
            },
            cache_folder="./.embedding_cache",  # Add caching for better performance
        )
        self.vector_store = None
        self.documents = []
        
    def __del__(self):
        """Clean up resources when the object is garbage collected."""
        self.cleanup()
        
    def cleanup(self):
        """Explicitly clean up resources to prevent memory and semaphore leaks."""
        try:
            # Clean up vector store if needed
            if hasattr(self, 'vector_store') and self.vector_store is not None:
                # Remove reference to help garbage collection
                self.vector_store = None
            
            # Clean up embeddings model
            if hasattr(self, 'embeddings') and self.embeddings is not None:
                # HuggingFaceEmbeddings don't have an explicit cleanup method,
                # but we can help garbage collection by removing the reference
                self.embeddings = None
                
            # Clear document references
            if hasattr(self, 'documents'):
                self.documents = []
                
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Cleaned up DocumentProcessor resources")
        except Exception as e:
            logger.error(f"Error during DocumentProcessor cleanup: {e}")

    def load_document(self, file_path: str):
        """
        Load a document from a file path.

        Args:
            file_path: Path to the document

        Returns:
            List of document objects
        """
        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".csv":
                loader = CSVLoader(file_path)
            elif file_extension in [".txt", ".md"]:
                loader = TextLoader(file_path)
            else:
                # For other file types, use a more general loader
                loader = UnstructuredFileLoader(file_path)

            self.documents = loader.load()
            return self.documents
        except Exception as e:
            raise ValueError(f"Error loading document: {str(e)}")

    def process_documents(self):
        """Process documents into chunks optimized for summarization."""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_document first.")

        return self.text_splitter.split_documents(self.documents)

    def create_vector_store(self):
        """
        Create a vector store from processed documents.
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_document first.")

        chunks = self.process_documents()
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)

    def similarity_search(self, query: str, k: int = 4):
        """
        Perform similarity search on the vector store.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of relevant document chunks
        """
        if not self.vector_store:
            raise ValueError(
                "Vector store not initialized. Call create_vector_store first."
            )

        return self.vector_store.similarity_search(query, k=k)

    def get_document_chunks(self, from_page: int, to_page: int) -> List[Document]:
        """
        Get document chunks from a specific page range.
        
        Args:
            from_page: Starting page number (inclusive, 0-indexed)
            to_page: Ending page number (exclusive, 0-indexed)
            
        Returns:
            List of document chunks in the specified range
            
        Raises:
            ValueError: If no documents are loaded or page numbers are invalid
        """
        if not self.documents:
            logger.error("No documents loaded. Call load_document first.")
            raise ValueError("No documents loaded. Call load_document first.")
        
        # Handle invalid page numbers
        total_pages = len(self.documents)
        
        # Validate and adjust page ranges
        from_page = max(0, min(from_page, total_pages - 1))
        to_page = max(from_page + 1, min(to_page + 1, total_pages))  # +1 to make it inclusive
        
        logger.info(f"Retrieving document chunks from page {from_page} to {to_page-1}")
        
        return self.documents[from_page:to_page]

    def _format_summary(self, chunks):
        """Format chunks into a readable summary"""
        if not chunks:
            return ""

        # Sort chunks by their order in the document if possible
        try:
            sorted_chunks = sorted(
                chunks,
                key=lambda x: (
                    x.metadata.get("page", 0),
                    x.metadata.get("chunk_id", 0),
                ),
            )
        except:
            sorted_chunks = chunks

        formatted_chunks = []
        for chunk in sorted_chunks:
            # Clean up text
            text = chunk.page_content.strip()
            if text:
                formatted_chunks.append(text)

        return "\n\n---\n\n".join(formatted_chunks)

    @time_execution
    def generate_abstractive_summarize(self, summary_chunks):
        """
        Create an abstractive summary using a language model.

        Args:
            summary_chunks: List of document chunks to summarize

        Returns:
            String with abstractive summary
        """
        try:
            print(f"Summarizing {len(summary_chunks)} chunks")

            # First combine the chunks to get a complete context
            if len(summary_chunks) <= 10:
                # For small documents, use stuff chain
                summarizer = pipeline(
                    "summarization",
                    model="google/gemma-3-1b-it",
                    device="mps",
                    max_length=120000,
                    min_length=50,
                    do_sample=True,
                    temperature=0.3,
                )

                llm = HuggingFacePipeline(pipeline=summarizer)

                # Use stuff chain for small documents
                map_reduce_chain = load_summarize_chain(
                    llm, chain_type="refine", verbose=True
                )

                result = map_reduce_chain.run(summary_chunks)
                return result
            else:
                # For larger documents, use map_reduce with custom prompts
                summarizer = pipeline(
                    "summarization",
                    model="meta-llama/Llama-3.2-3B-Instruct",
                    device="mps",
                    max_length=32000,
                    min_length=30,
                    do_sample=True,
                    temperature=0.1,
                    repetition_penalty=1.3,
                    length_penalty=0.8,  # Favor shorter outputs
                )

                llm = HuggingFacePipeline(pipeline=summarizer)

                # Use map_reduce with custom prompts
                print("Using map_reduce chain with custom prompts")
                summary_chain = load_summarize_chain(
                    llm, chain_type="map_reduce", verbose=True
                )

                result = summary_chain.run(summary_chunks)

                print(f"Final summary: {result}")
                return result

        except Exception as e:
            # Fallback to extractive if abstractive fails
            print(f"Abstractive summarization failed: {e}")
            import traceback

            traceback.print_exc()
            return self._format_summary(summary_chunks)

    @time_execution
    def generate_brief_summary(self, summary_chunks) -> str:
        """
        Creates a concise ~150 word summary of document chunks using Gemma.

        Args:
            summary_chunks: List of document chunks

        Returns:
            str: Brief summary of content
        """
        try:
            # Combine text from all chunks with proper spacing
            combined_text = "\n\n".join(
                [chunk.page_content for chunk in summary_chunks]
            )

            # Create a summarization prompt
            prompt = f"""Summarize the following text in about 150 words.
Focus only on the main concepts, key points, and central ideas:

{combined_text}

Summary:"""

            # Initialize the model for text generation
            text_generation = pipeline(
                "text-generation",
                model="google/gemma-3-1b-it",
                device="mps",
                torch_dtype=torch.float16,
                do_sample=True,
                temperature=0.3,  # Slightly higher for more natural language
                max_new_tokens=300,  # Allow enough tokens for ~200 words
                min_new_tokens=150,  # Ensure at least ~100 words
                repetition_penalty=1.2,  # Prevent repetitive text
                return_full_text=False,  # Only return the generated text
            )

            # Format prompt for instruction-tuned model
            formatted_prompt = (
                f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            )

            # Generate the summary
            outputs = text_generation(formatted_prompt)
            if not outputs or not isinstance(outputs, list) or len(outputs) == 0:
                raise ValueError("No output generated from the model")

            generated_text = outputs[0].get("generated_text", "")
            if not generated_text:
                raise ValueError("No text generated from the model")

            # Clean up any special tokens and ensure reasonable length
            summary = generated_text.replace("<end_of_turn>", "").strip()

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}", exc_info=True)
            return f"Error generating summary: {str(e)}"

    def get_document_summary(self):
        chunks = self.process_documents()
        return self.generate_abstractive_summarize(chunks)
