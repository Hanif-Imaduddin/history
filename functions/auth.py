"""JWT token creation/verification and password hashing utilities."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Cookie, Depends, HTTPException
from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "changeme-please-set-a-random-secret")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24
DAILY_ANALYSIS_LIMIT = 5

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_token(user_id: int, username: str, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def get_current_user(access_token: Optional[str] = Cookie(default=None)) -> dict:
    """FastAPI dependency — raises 401 if not authenticated."""
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = _decode_token(access_token)
        return {
            "user_id": int(payload["sub"]),
            "username": payload["username"],
            "role": payload["role"],
        }
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """FastAPI dependency — raises 403 if not admin."""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def get_user_optional(access_token: Optional[str] = Cookie(default=None)) -> Optional[dict]:
    """Soft auth check — returns None instead of raising 401."""
    if not access_token:
        return None
    try:
        payload = _decode_token(access_token)
        return {
            "user_id": int(payload["sub"]),
            "username": payload["username"],
            "role": payload["role"],
        }
    except JWTError:
        return None
