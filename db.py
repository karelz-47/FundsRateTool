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
from sqlalchemy.sql import func


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
    else:
        # local fallback
        db_url = "sqlite:///./local.db"
    return create_engine(db_url, pool_pre_ping=True)


class Base(DeclarativeBase):
    pass


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


class FxRate(Base):
    __tablename__ = "fx_rate"
    __table_args__ = (UniqueConstraint("rate_date", name="uq_fx_rate_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rate_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    huf_buy: Mapped[float] = mapped_column(Float, nullable=False)
    huf_mid: Mapped[float] = mapped_column(Float, nullable=False)
    usd_sell: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class FundNav(Base):
    __tablename__ = "fund_nav"
    __table_args__ = (UniqueConstraint("nav_date", "isin", name="uq_nav_date_isin"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nav_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    isin: Mapped[str] = mapped_column(String(16), nullable=False, index=True)

    nav: Mapped[float] = mapped_column(Float, nullable=False)
    ccy: Mapped[str] = mapped_column(String(3), nullable=False)

    fund_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    source_email_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    raw_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class CalcRun(Base):
    __tablename__ = "calc_run"
    __table_args__ = (UniqueConstraint("input_hash", name="uq_calc_run_input_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

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
    __table_args__ = (
        UniqueConstraint("rate_date", "series_code", name="uq_published_rates_date_code"),
        Index("ix_published_rates_rate_date", "rate_date"),
        Index("ix_published_rates_series_code", "series_code"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rate_date: Mapped[date] = mapped_column(Date, nullable=False)
    series_code: Mapped[str] = mapped_column(String(32), nullable=False)  # TR_HUF, ISIN, CONSERVATIVE...
    value: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="xlsm_backfill")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def init_db():
    eng = get_engine()
    Base.metadata.create_all(bind=eng)


def compute_input_hash(payload: Dict[str, Any]) -> str:
    b = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(b).hexdigest()
