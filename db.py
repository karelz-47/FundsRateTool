from __future__ import annotations

import os
import json
import hashlib
from datetime import datetime, date
from typing import Any, Dict, Optional

from sqlalchemy import (
    create_engine,
    String,
    Integer,
    Date,
    DateTime,
    Float,
    Text,
    ForeignKey,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def _normalize_db_url(url: str) -> str:
    """
    Railway sometimes provides postgres:// which SQLAlchemy treats as postgresql://.
    Also force psycopg v3 driver (avoids psycopg2 dependency).
    """
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)

    return url


def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        db_url = _normalize_db_url(db_url)
        return create_engine(db_url, pool_pre_ping=True)

    # local fallback (SQLite)
    return create_engine("sqlite:///app.db", connect_args={"check_same_thread": False})


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


class Base(DeclarativeBase):
    pass


class FxRate(Base):
    """
    Tatra banka FX rates imported from CSV.
    IMPORTANT: keep table name stable to match existing DB.
    """
    __tablename__ = "fx_rates_tb"
    __table_args__ = (
        UniqueConstraint("rate_date", name="uq_fx_rate_date"),
        Index("ix_fx_rate_date", "rate_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rate_date: Mapped[date] = mapped_column(Date, nullable=False)

    # HUF per EUR (buy/sell/mid) or as you store them from TB CSV
    huf_buy: Mapped[float] = mapped_column(Float, nullable=False)
    huf_sell: Mapped[float] = mapped_column(Float, nullable=False)
    huf_mid: Mapped[float] = mapped_column(Float, nullable=False)

    # USD per EUR (buy/sell/mid) or TB-provided USD legs
    usd_buy: Mapped[float] = mapped_column(Float, nullable=False)
    usd_sell: Mapped[float] = mapped_column(Float, nullable=False)
    usd_mid: Mapped[float] = mapped_column(Float, nullable=False)

    source: Mapped[str] = mapped_column(String(32), default="csv_import")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FundNav(Base):
    """
    NAV values parsed from pasted email text (baha.com notification).
    """
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

    # for audit / debugging
    source_email_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    raw_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CashAlloc(Base):
    """
    Kept for backward compatibility (even if UI removed).
    If you no longer use this table, you can later delete it via a migration.
    """
    __tablename__ = "cash_alloc"
    __table_args__ = (
        UniqueConstraint("alloc_date", "series_code", name="uq_cash_alloc_date_series"),
        Index("ix_cash_alloc_date", "alloc_date"),
        Index("ix_cash_alloc_series", "series_code"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alloc_date: Mapped[date] = mapped_column(Date, nullable=False)
    series_code: Mapped[str] = mapped_column(String(32), nullable=False)
    cash_pct: Mapped[float] = mapped_column(Float, nullable=False)

    source: Mapped[str] = mapped_column(String(32), default="manual")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CalcRun(Base):
    __tablename__ = "calc_run"
    __table_args__ = (
        UniqueConstraint("input_hash", name="uq_calc_run_input_hash"),
        Index("ix_calc_run_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    params_json: Mapped[str] = mapped_column(Text, nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    rows: Mapped[list["CalcDaily"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class CalcDaily(Base):
    __tablename__ = "calc_daily"
    __table_args__ = (
        UniqueConstraint("run_id", "calc_date", name="uq_run_date"),
        Index("ix_calc_daily_date", "calc_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("calc_run.id"), nullable=False)
    calc_date: Mapped[date] = mapped_column(Date, nullable=False)

    output_json: Mapped[str] = mapped_column(Text, nullable=False)

    run: Mapped["CalcRun"] = relationship(back_populates="rows")


class PublishedRate(Base):
    """
    Canonical “published history” of NOVIS internal fund rates (backfilled from legacy XLSM
    + appended with computed days after watermark).
    """
    __tablename__ = "published_rates"
    __table_args__ = (
        UniqueConstraint("rate_date", "series_code", name="uq_published_rates_date_code"),
        Index("ix_published_rates_date", "rate_date"),
        Index("ix_published_rates_series", "series_code"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rate_date: Mapped[date] = mapped_column(Date, nullable=False)
    series_code: Mapped[str] = mapped_column(String(32), nullable=False)  # TR_HUF, ISIN, CONSERVATIVE...
    value: Mapped[float] = mapped_column(Float, nullable=False)

    source: Mapped[str] = mapped_column(String(64), nullable=False, default="xlsm_backfill")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def init_db():
    eng = get_engine()
    # checkfirst=True is default; it will not recreate existing tables.
    Base.metadata.create_all(bind=eng)


def compute_input_hash(payload: Dict[str, Any]) -> str:
    b = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(b).hexdigest()
