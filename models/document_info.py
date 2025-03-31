from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DocumentInfo:
    """Document information model for the AI Tutor, distinct from LangChain's Document."""
    id: Optional[str] = None
    title: str = "Untitled Document"
    filename: str = "unknown"
    session_id: str = ""
    author: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    file_size: int = 0
    chunk_count: Optional[int] = None
    created_at: Optional[datetime] = None
    
    @staticmethod
    def from_dict(data: Dict) -> "DocumentInfo":
        """Convert a dictionary to DocumentInfo object."""
        return DocumentInfo(
            id=data.get("_id") or data.get("id"),
            title=data.get("title", "Untitled Document"),
            filename=data.get("filename", "unknown"),
            session_id=data.get("session_id", ""),
            author=data.get("author"),
            description=data.get("description"),
            tags=data.get("tags", []),
            file_size=data.get("file_size", 0),
            chunk_count=data.get("chunk_count"),
            created_at=data.get("created_at"),
        )
    
    def to_dict(self) -> Dict:
        """Convert DocumentInfo to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "filename": self.filename,
            "session_id": self.session_id,
            "author": self.author,
            "description": self.description,
            "tags": self.tags,
            "file_size": self.file_size,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }