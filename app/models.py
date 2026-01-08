# app/models.py
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class Asset(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True, unique=True)
    name: Optional[str] = None


class NewsItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)               # e.g., "AAPL" (or store asset_id instead)
    source: Optional[str] = None
    title: str
    body: Optional[str] = None
    url: Optional[str] = Field(default=None, unique=True)
    published_at: datetime = Field(index=True)


class PriceBar(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    ts: datetime = Field(index=True)
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class Prediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    ts: datetime = Field(index=True)

    model_version: str = Field(index=True)
    prob_up: float
    prob_down: float
    created_at: datetime = Field(default_factory=datetime.now(), index=True)