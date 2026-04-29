# app/main.py
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

from fastapi import FastAPI, Depends, HTTPException
from app.routers import health
from sqlmodel import SQLModel, Session, select

from app.db import engine, get_session
from app.models import Asset, NewsItem

app = FastAPI()
app.include_router(health.router)

@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    yield
    # Place for shutdown code if needed

