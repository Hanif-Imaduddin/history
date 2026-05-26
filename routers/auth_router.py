"""Authentication routes: register, login, logout, me."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Cookie, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from functions.auth import create_token, hash_password, verify_password, _decode_token
from functions.postgres import (
    add_fingerprint,
    check_fingerprint,
    create_user,
    get_user_by_username,
    log_auth_event,
)
from jose import JWTError

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str
    password: str
    confirm_password: str
    fingerprint: str = ""


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
async def register(req: RegisterRequest, request: Request, response: Response):
    username = req.username.strip()
    if len(username) < 3 or len(username) > 50:
        raise HTTPException(400, "Username harus 3–50 karakter.")
    if len(req.password) < 6:
        raise HTTPException(400, "Password minimal 6 karakter.")
    if req.password != req.confirm_password:
        raise HTTPException(400, "Password dan konfirmasi password tidak cocok.")
    if await run_in_threadpool(get_user_by_username, username):
        raise HTTPException(409, "Username sudah dipakai.")

    ip = request.client.host if request.client else ""

    # Block registration if fingerprint already belongs to another user
    if req.fingerprint:
        existing_uid = await run_in_threadpool(check_fingerprint, req.fingerprint)
        if existing_uid:
            await run_in_threadpool(
                log_auth_event,
                None, username, "register_fp_duplicate",
                f"fingerprint already used by user_id={existing_uid}", ip,
            )
            raise HTTPException(
                409,
                "Perangkat ini sudah terdaftar dengan akun lain. Gunakan akun yang sudah ada atau hubungi admin.",
            )

    user_id = await run_in_threadpool(create_user, username, hash_password(req.password), "user")

    if req.fingerprint:
        await run_in_threadpool(add_fingerprint, user_id, req.fingerprint)

    await run_in_threadpool(log_auth_event, user_id, username, "register", "", ip)

    token = create_token(user_id, username, "user")
    response.set_cookie("access_token", token, httponly=True, samesite="lax", max_age=86400)
    return {"message": "Registrasi berhasil.", "username": username, "role": "user"}


@router.post("/login")
async def login(req: LoginRequest, request: Request, response: Response):
    ip = request.client.host if request.client else ""
    user = await run_in_threadpool(get_user_by_username, req.username.strip())

    if not user or not verify_password(req.password, user["password_hash"]):
        await run_in_threadpool(log_auth_event, None, req.username, "login_failed", "", ip)
        raise HTTPException(401, "Username atau password salah.")
    if not user["is_active"]:
        raise HTTPException(403, "Akun ini telah dinonaktifkan. Hubungi admin.")

    await run_in_threadpool(log_auth_event, user["id"], user["username"], "login", "", ip)
    token = create_token(user["id"], user["username"], user["role"])
    response.set_cookie("access_token", token, httponly=True, samesite="lax", max_age=86400)
    return {"message": "Login berhasil.", "username": user["username"], "role": user["role"]}


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"message": "Logged out."}


@router.get("/me")
async def get_me(access_token: Optional[str] = Cookie(default=None)):
    """Soft auth check — returns authenticated: false instead of raising 401."""
    if not access_token:
        return {"authenticated": False}
    try:
        payload = _decode_token(access_token)
        return {
            "authenticated": True,
            "user_id": int(payload["sub"]),
            "username": payload["username"],
            "role": payload["role"],
        }
    except JWTError:
        return {"authenticated": False}
