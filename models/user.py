from pydantic import BaseModel
from typing import Optional, Dict


class User(BaseModel):
    """User model with basic information."""

    id: Optional[str] = None
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None

    class Config:
        # Allow MongoDB ObjectID conversion
        arbitrary_types_allowed = True

class UserInDB(User):
    """User model with password hash."""

    hashed_password: str


class Token(BaseModel):
    """Token response model."""

    access_token: str
    token_type: str


class UserCreate(BaseModel):
    """User creation model."""

    username: str
    password: str
    email: Optional[str] = None
    full_name: Optional[str] = None
