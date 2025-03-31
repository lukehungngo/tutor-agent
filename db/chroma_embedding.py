import os
from typing import List, Dict
from datetime import datetime, timezone
from pymongo import MongoClient
from bson.objectid import ObjectId
import chromadb
from langchain.schema import Document
from utils import logger, time_execution
from config.settings import settings

class ChromaEmbeddingStore:
    """
    Hybrid storage system that uses MongoDB for document metadata
    and ChromaDB for vector embeddings, with persistence capabilities.
    """
    
    def __init__(self, collection_name: str = "embeddings", persist_directory: str = "./chroma_db"):
        """Initialize the ChromaDB embedding store.
        
        Args:
            collection_name: MongoDB collection name for document metadata
            persist_directory: Directory to store ChromaDB files
        """
        # MongoDB for document metadata
        self.client = MongoClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_collection_name]
        self.collection = self.db[collection_name]
        
        # Local file paths for ChromaDB persistence
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with the default embedding function
        # ChromaDB now includes built-in HuggingFace embedding support
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Track ChromaDB collections by session
        self.collections = {}
        
        logger.info(f"Initialized ChromaEmbeddingStore with persistence at {self.persist_directory}")
    
    def _get_collection_name(self, session_id: str) -> str:
        """Get the standardized collection name for a session."""
        return f"session_{session_id}"
    
    @time_execution
    def create_session(self, session_id: str, documents: List[Document]) -> List[str]:
        """Create a new session with documents and save embeddings.
        
        Args:
            session_id: Unique session identifier
            documents: List of documents to embed and store
            
        Returns:
            List of MongoDB document IDs
        """
        collection_name = self._get_collection_name(session_id)
        
        # Create or get the ChromaDB collection
        try:
            # Try to get existing collection first
            collection = self.chroma_client.get_collection(
                name=collection_name
            )
            # If collection exists, delete all documents
            collection.delete(collection.get()["ids"])
        except:
            # Create new collection if it doesn't exist
            collection = self.chroma_client.create_collection(
                name=collection_name,
                # Uses the default HuggingFace embedding function
                # Default in ChromaDB is "all-MiniLM-L6-v2"
            )
        
        # Store reference to collection
        self.collections[session_id] = collection
        
        # Store documents in MongoDB and prepare for ChromaDB
        document_ids = []
        chroma_ids = []
        chroma_docs = []
        chroma_metadatas = []
        
        for i, doc in enumerate(documents):
            # Store in MongoDB
            metadata = doc.metadata.copy() if hasattr(doc, 'metadata') and doc.metadata else {}
            metadata.update({
                "session_id": session_id, 
                "chunk_id": i,
                "created_at": datetime.now(timezone.utc).isoformat()  # Use string format for ChromaDB
            })
            
            result = self.collection.insert_one({
                "text": doc.page_content,
                "metadata": metadata
            })
            
            mongo_id = str(result.inserted_id)
            document_ids.append(mongo_id)
            
            # Prepare data for ChromaDB
            chroma_id = f"{session_id}_{i}"
            chroma_ids.append(chroma_id)
            chroma_docs.append(doc.page_content)
            
            # Include MongoDB ID in the metadata
            # Filter out non-serializable types for ChromaDB
            chroma_metadata = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}
            chroma_metadata["mongo_id"] = mongo_id
            chroma_metadatas.append(chroma_metadata)
        
        # Add documents to ChromaDB
        collection.add(
            ids=chroma_ids,
            documents=chroma_docs,
            metadatas=chroma_metadatas
        )
        
        logger.info(f"Created session {session_id} with {len(documents)} documents in ChromaDB")
        return document_ids
    
    @time_execution
    def load_session(self, session_id: str) -> bool:
        """Load a session from persistent storage.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            True if session was loaded successfully, False otherwise
        """
        # Check if already loaded
        if session_id in self.collections:
            return True
        
        collection_name = self._get_collection_name(session_id)
        
        # Try to get collection from ChromaDB
        try:
            collection = self.chroma_client.get_collection(
                name=collection_name
            )
            self.collections[session_id] = collection
            logger.info(f"Loaded session {session_id} from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error loading session {session_id} from ChromaDB: {e}")
            return False
    
    @time_execution
    def similarity_search(self, session_id: str, query: str, k: int = 4) -> List[Dict]:
        """Perform similarity search for a query.
        
        Args:
            session_id: Session ID for the search
            query: Query text
            k: Number of results to return
            
        Returns:
            List of search results with text and metadata
        """
        # Load session if not already loaded
        if session_id not in self.collections:
            success = self.load_session(session_id)
            if not success:
                logger.error(f"Failed to load session {session_id}")
                return []
        
        # Get ChromaDB collection
        collection = self.collections[session_id]
        
        # Perform similarity search
        results = collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Format results
        formatted_results = []
        
        if not results or not results["ids"] or len(results["ids"][0]) == 0:
            logger.warning(f"No results found for query in session {session_id}")
            return []
        
        for i in range(len(results["ids"][0])):
            # Get metadata from ChromaDB result
            chroma_metadata = results["metadatas"][0][i]
            mongo_id = chroma_metadata.get("mongo_id")
            
            # Create base result with ChromaDB data
            result = {
                "text": results["documents"][0][i],
                "metadata": chroma_metadata,
                "score": results["distances"][0][i] if "distances" in results else None
            }
            
            # Fetch full document from MongoDB if ID is available
            if mongo_id:
                try:
                    db_doc = self.collection.find_one({"_id": ObjectId(mongo_id)})
                    if db_doc:
                        # Use MongoDB as source of truth for text
                        result["text"] = db_doc["text"]
                        # Merge metadata, prioritizing MongoDB metadata
                        result["metadata"] = db_doc.get("metadata", {})
                except Exception as e:
                    logger.warning(f"Error fetching document {mongo_id} from MongoDB: {e}")
            
            formatted_results.append(result)
        
        return formatted_results
    
    def get_all_documents(self, session_id: str) -> List[Dict]:
        """Get all documents for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of documents with text and metadata
        """
        # Query MongoDB for all documents in this session
        documents = self.collection.find({"metadata.session_id": session_id})
        return list(documents)
    
    @time_execution
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its data.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Remove from memory
            if session_id in self.collections:
                del self.collections[session_id]
            
            # Delete collection from ChromaDB
            collection_name = self._get_collection_name(session_id)
            try:
                self.chroma_client.delete_collection(collection_name)
            except Exception as e:
                logger.warning(f"Error deleting ChromaDB collection: {e}")
            
            # Delete documents from MongoDB
            self.collection.delete_many({"metadata.session_id": session_id})
            
            logger.info(f"Deleted session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Clear in-memory collections
            self.collections = {}
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Cleaned up ChromaEmbeddingStore resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")