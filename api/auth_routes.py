from fastapi import APIRouter, Depends, HTTPException, status
from services import auth_service
from models import User, Token
from pydantic import BaseModel
from utils import logger


# Create model for login request
class LoginRequest(BaseModel):
    username: str
    password: str


router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=Token)
async def login(request: LoginRequest):
    """Login endpoint that also handles first-time registration."""
    result = await auth_service.authenticate_user(request.username, request.password)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    logger.info(
        f"User logged in successfully, username: {request.username}, access token: {result[1]}"
    )
    user, access_token = result
    return Token(access_token=access_token, token_type="bearer")


@router.get("/profile", response_model=User)
async def get_current_user(user: User = Depends(auth_service.require_auth)):
    """Get the currently authenticated user."""
    return user
