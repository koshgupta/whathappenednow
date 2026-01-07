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

@app.post("/assets", response_model=Asset)
def create_asset(asset: Asset, session: Session = Depends(get_session)):
    # Enforce unique symbol at app level (SQLite unique constraint also helps)
    existing = session.exec(select(Asset).where(Asset.symbol == asset.symbol)).first()
    if existing:
        raise HTTPException(status_code=409, detail="Symbol already exists")

    session.add(asset)
    session.commit()
    session.refresh(asset)
    return asset

@app.get("/assets", response_model=List[Asset])
def list_assets(session: Session = Depends(get_session)):
    return session.exec(select(Asset).order_by(Asset.symbol)).all()

@app.post("/news", response_model=NewsItem)
def ingest_news(item: NewsItem, session: Session = Depends(get_session)):
    # Basic dedupe by URL if provided
    if item.url:
        existing = session.exec(select(NewsItem).where(NewsItem.url == item.url)).first()
        if existing:
            return existing

    session.add(item)
    session.commit()
    session.refresh(item)
    return item

@app.get("/news/{symbol}", response_model=List[NewsItem])
def get_news(symbol: str, session: Session = Depends(get_session)):
    stmt = select(NewsItem).where(NewsItem.symbol == symbol.upper()).order_by(NewsItem.published_at.desc())
    return session.exec(stmt).all()
