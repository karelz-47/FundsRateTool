from __future__ import annotations

import os
import json
import hashlib
from datetime import datetime, date
from typing import Any, Dict, Optional

from sqlalchemy import (
    create_engine, String, Integer, Date, DateTime, Float, Text, ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def _normalize_db_url(url: str) -> str:
    # Railway sometimes provides postgres:// which SQLAlchemy accepts as postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # Force SQLAlchemy to use psycopg v3 (NOT psycopg2)
    # This avoids: ModuleNotFoundError: No module named 'psycopg2'
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)

    return url


def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        db_url = _normalize_db_url(db_url)
        return create_engine(db_url, pool_pre_ping=True)
    return create_engine("sqlite:///app.db", connect_args={"check_same_thread": False})


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


class Base(DeclarativeBase):
    pass


class FxRate(Base):
    __tablename__ = "fx_rates_tb"
    __table_args__ = (UniqueConstraint("rate_date", name="uq_fx_rate_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rate_date: Mapped[date] = mapped_column(Date, nullable=False)

    huf_buy: Mapped[float] = mapped_column(Float, nullable=False)
    huf_sell: Mapped[float] = mapped_column(Float, nullable=False)
    huf_mid: Mapped[float] = mapped_column(Float, nullable=False)

    usd_buy: Mapped[float] = mapped_column(Float, nullable=False)
    usd_sell: Mapped[float] = mapped_column(Float, nullable=False)
    usd_mid: Mapped[float] = mapped_column(Float, nullable=False)

    source: Mapped[str] = mapped_column(String(32), default="csv_import")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FundNav(Base):
    __tablename__ = "fund_nav"
    __table_args__ = (
        UniqueConstraint("nav_date", "isin", name="uq_nav_date_isin"),
        Index("ix_nav_date", "nav_date"),
        Index("ix_nav_isin", "isin"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nav_date: Mapped[date] = mapped_column(Date, nullable=False)
    isin: Mapped[str] = mapped_column(String(16), nullable=False)

    nav: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    fund_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    source: Mapped[str] = mapped_column(String(32), default="paste")
    raw_excerpt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CashAlloc(Base):
    __tablename__ = "cash_alloc"
    __table_args__ = (UniqueConstraint("alloc_date", "series_code", name="uq_alloc_date_series"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alloc_date: Mapped[date] = mapped_column(Date, nullable=False)
    series_code: Mapped[str] = mapped_column(String(32), nullable=False)
    cash_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # 0..1

    source: Mapped[str] = mapped_column(String(32), default="manual")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CalcRun(Base):
    __tablename__ = "calc_run"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    rows: Mapped[list["CalcDaily"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class CalcDaily(Base):
    __tablename__ = "calc_daily"
    __table_args__ = (UniqueConstraint("run_id", "calc_date", name="uq_run_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("calc_run.id"), nullable=False)
    calc_date: Mapped[date] = mapped_column(Date, nullable=False)

    output_json: Mapped[str] = mapped_column(Text, nullable=False)

    run: Mapped["CalcRun"] = relationship(back_populates="rows")

class PublishedRate(Base):
    __tablename__ = "published_rates"
    id = Column(Integer, primary_key=True, index=True)
    rate_date = Column(Date, nullable=False, index=True)
    series_code = Column(String(32), nullable=False, index=True)  # TR_HUF, ISIN, CONSERVATIVE...
    value = Column(Float, nullable=False)
    source = Column(String(64), nullable=False, default="xlsm_backfill")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("rate_date", "series_code", name="uq_published_rates_date_code"),
    )

def init_db():
    eng = get_engine()
    Base.metadata.create_all(bind=eng)


def compute_input_hash(payload: Dict[str, Any]) -> str:
    b = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


