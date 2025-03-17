from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


class VectorStoreManager:
    """Manages vector storage and retrieval operations."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

    def create_vector_store(self, documents: List[Document]) -> None:
        """Create a new vector store from documents."""
        self.vector_store = FAISS.from_documents(
            documents=documents, embedding=self.embeddings
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search."""
        if not self.vector_store:
            raise ValueError(
                "Vector store not initialized. Call create_vector_store first."
            )

        return self.vector_store.similarity_search(query, k=k)
