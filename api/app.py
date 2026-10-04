"""Run locally: python -m uvicorn api.app:app --host 127.0.0.1 --port 8011"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from api.service import ModelService, PROFILES


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_request: str = Field(min_length=2, max_length=500)
    profile: Literal["all", "query", "calculate", "calendar"] = "all"


class ConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmation_token: str = Field(pattern=r"^[0-9a-f]{32}$")


class RouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_request: str = Field(min_length=2, max_length=500)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = ModelService()
    yield
    del app.state.service


app = FastAPI(title="中文工具调用本地推理服务", version="0.1.0", lifespan=lifespan)
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(request: Request):
    service = request.app.state.service
    return {"status": "ok", "model": service.model_name, "model_revision": service.model_revision,
            "adapter_sha256": service.adapter_hash,
            "route_adapter_sha256": service.route_adapter_hash, "profiles": PROFILES}


@app.post("/decide")
def decide(body: DecideRequest, request: Request):
    return request.app.state.service.decide(body.user_request, body.profile)


@app.post("/route")
def route(body: RouteRequest, request: Request):
    return request.app.state.service.route(body.user_request)


@app.post("/confirm")
def confirm(body: ConfirmRequest, request: Request):
    result = request.app.state.service.confirm(body.confirmation_token)
    if result is None:
        raise HTTPException(status_code=404, detail="待确认操作不存在、已使用或已过期")
    return result
