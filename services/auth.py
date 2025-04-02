from datetime import datetime, timedelta, timezone
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Optional, Tuple
import hashlib
import os
from config.settings import settings
from utils import logger
from models import User
from db import UserRepository


# Simple hashing for passwords
def hash_password(password: str) -> str:
    """Hash a password with a random salt."""
    salt = os.urandom(32)  # 32 bytes of random data
    key = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, 100000  # Number of iterations
    )
    return salt.hex() + ":" + key.hex()


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash."""
    try:
        salt_hex, key_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            100000,  # Same number of iterations as in hash_password
        )
        return key.hex() == key_hex
    except Exception:
        return False


# OAuth2 scheme for token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


class AuthService:
    def __init__(self):
        self.user_repository = UserRepository()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a hash."""
        return verify_password(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Generate a hash from a password."""
        return hash_password(password)

    def create_access_token(self, username: str) -> str:
        """Create a JWT token."""
        to_encode = {
            "sub": username,
            "exp": datetime.now(timezone.utc)
            + timedelta(minutes=settings.JWT_EXPIRATION_TIME_MINUTES),
        }
        encoded_jwt = jwt.encode(
            to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )
        return encoded_jwt

    async def authenticate_user(
        self, username: str, password: str
    ) -> Optional[Tuple[User, str]]:
        """Authenticate user - register if first login."""
        # Check if user exists
        user_dict = self.user_repository.db.users.find_one({"username": username})

        if not user_dict:
            # First login - register user
            hashed_password = self.get_password_hash(password)
            user_dict = {
                "username": username,
                "hashed_password": hashed_password,
                "email": None,
                "full_name": None,
            }
            result = self.user_repository.db.users.insert_one(user_dict)
            user_dict["id"] = str(result.inserted_id)
            logger.info(f"New user registered successfully: {username}")
        else:
            # Verify password for existing user
            if not self.verify_password(password, user_dict.get("hashed_password", "")):
                return None
            logger.info(f"User logged in successfully: {username}")

        # Create access token
        access_token = self.create_access_token(username)

        # Return user without hashed password and token
        user = User(
            id=str(user_dict.get("_id", user_dict.get("id", ""))),
            username=user_dict["username"],
            email=user_dict.get("email"),
            full_name=user_dict.get("full_name"),
        )
        return user, access_token

    async def get_current_user(
        self, token: Optional[str] = Depends(oauth2_scheme)
    ) -> Optional[User]:
        """Get current user from token."""
        if not token:
            return None

        try:
            payload = jwt.decode(
                token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
            )
            username: str = payload.get("sub")
        except jwt.PyJWTError as e:
            logger.error(f"JWT error: {e}")
            return None

        return self.user_repository.get_user(username)

    async def require_auth(self, token: Optional[str] = Depends(oauth2_scheme)) -> User:
        """Require authentication - use as dependency."""
        user = await self.get_current_user(token)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        assert user.id is not None, "User ID cannot be None"  # Assure the type checker
        return user
