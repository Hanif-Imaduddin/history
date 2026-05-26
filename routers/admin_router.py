"""Admin-only routes: user management and logs."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from functions.auth import hash_password, require_admin
from functions.postgres import (
    add_fingerprint,
    create_user,
    delete_user,
    get_analysis_logs,
    get_auth_logs,
    get_user_by_username,
    list_all_users,
)

router = APIRouter(prefix="/admin", tags=["admin"])


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


@router.get("/users")
async def admin_list_users(current_user: dict = Depends(require_admin)):
    return await run_in_threadpool(list_all_users)


@router.post("/users")
async def admin_create_user(
    req: CreateUserRequest,
    current_user: dict = Depends(require_admin),
):
    username = req.username.strip()
    if len(username) < 3:
        raise HTTPException(400, "Username minimal 3 karakter.")
    if len(req.password) < 6:
        raise HTTPException(400, "Password minimal 6 karakter.")
    if req.role not in ("user", "admin"):
        raise HTTPException(400, "Role tidak valid. Pilih 'user' atau 'admin'.")
    if await run_in_threadpool(get_user_by_username, username):
        raise HTTPException(409, "Username sudah dipakai.")

    user_id = await run_in_threadpool(create_user, username, hash_password(req.password), req.role)
    return {"message": "Akun berhasil dibuat.", "user_id": user_id, "username": username, "role": req.role}


@router.delete("/users/{user_id}")
async def admin_delete_user(user_id: int, current_user: dict = Depends(require_admin)):
    if user_id == current_user["user_id"]:
        raise HTTPException(400, "Tidak bisa menghapus akun sendiri.")
    if not await run_in_threadpool(delete_user, user_id):
        raise HTTPException(404, "User tidak ditemukan.")
    return {"message": "Akun berhasil dihapus."}


@router.get("/logs/analysis")
async def admin_analysis_logs(current_user: dict = Depends(require_admin)):
    return await run_in_threadpool(get_analysis_logs, 200)


@router.get("/logs/auth")
async def admin_auth_logs(current_user: dict = Depends(require_admin)):
    return await run_in_threadpool(get_auth_logs, 200)
