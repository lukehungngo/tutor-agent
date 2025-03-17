from typing import List, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PDFLoader, UnstructuredFileLoader
import os


class DocumentProcessor:
    """Handles document loading and chunking for the tutor agent."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def load_document(self, file_path: str) -> List[Document]:
        """Load a document from file path."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                loader = PDFLoader(file_path)
            elif file_extension in ['.txt', '.md']:
                loader = TextLoader(file_path)
            else:
                # For other file types, use a more general loader
                loader = UnstructuredFileLoader(file_path)
                
            return loader.load()
        except Exception as e:
            raise ValueError(f"Error loading document: {str(e)}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)

    def process_with_summary(self, documents: List[Document], summary: Optional[str] = None) -> List[Document]:
        """Process documents with optional summary."""
        chunks = self.split_documents(documents)
        
        # If summary is provided, add it as a special document with high importance
        if summary and summary.strip():
            summary_doc = Document(
                page_content=summary,
                metadata={"source": "user_summary", "importance": "high"}
            )
            # Add summary at the beginning for higher priority
            chunks.insert(0, summary_doc)
            
        return chunks
