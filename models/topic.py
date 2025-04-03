from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DocumentInfo:
    """Document information model for the AI Tutor, distinct from LangChain's Document."""

    id: Optional[str] = None
    user_id: Optional[str] = None
    title: str = "Untitled Document"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None

    @staticmethod
    def from_dict(data: Dict) -> "DocumentInfo":
        """Convert a dictionary to DocumentInfo object."""
        return DocumentInfo(
            id=data.get("_id") or data.get("id"),
            user_id=data.get("user_id", None),
            title=data.get("title", "Untitled Document"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            created_at=data.get("created_at"),
        )

    def to_dict(self) -> Dict:
        """Convert DocumentInfo to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
