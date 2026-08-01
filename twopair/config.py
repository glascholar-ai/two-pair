"""Strategy and system configuration.

A single frozen dataclass carries every tunable. Defaults reproduce the
baseline v3 backtest exactly; live-only fields (symbols, notional, paths,
credentials) sit alongside so one object configures the whole system.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
from typing import Any, Optional


@dataclasses.dataclass(frozen=True)
class Config:
    """All strategy, risk, and system parameters."""

    # Instruments (KR leg / US leg of the pair).
    kr_symbol: str = "SKHYNIXUSDT"
    us_symbol: str = "SKHYUSDT"

    # Signal parameters (baseline v3 — see README.md before changing).
    win_mu: int = 288          # anchor window, 5m bars; keep at 24h multiples
    min_mu: int = 144
    win_sd: int = 300          # same-segment std window; safe zone 300-450
    min_sd: int = 100
    z_in: float = 2.0
    z_out: float = 0.5
    max_hold_hours: float = 24.0
    mtm_stop_pct: float = 2.5  # 0 disables
    bar_seconds: int = 300

    # Risk parameters.
    z_alert: float = 4.5
    daily_loss_halt_pct: float = 3.0     # halt new entries after this daily loss
    max_data_gap_bars: int = 2           # stale-data guard
    fx_stale_max_minutes: float = 120.0  # weekday FX staleness limit

    # Per-cycle position sync (plan A): exchange is the source of truth.
    sync_tolerance_pct: float = 5.0      # max leg-notional mismatch to accept
    dust_usdt: float = 10.0              # below this a leg counts as flat
    rearm_recovery_hours: float = 6.0    # startup: last stop within N hours
                                         # re-latches the re-arm suppressor

    # Sizing: single-leg notional in USDT (lambda relative to equity is the
    # operator's concern; the engine only knows absolute notional).
    leg_notional_usdt: float = 1000.0

    # Execution style. "bbo": post-only limit joining the touch (BUY at best
    # bid / SELL at best ask), re-quoted on timeout; falls back to a market
    # order after max_chases. "market": immediate market orders.
    order_style: str = "bbo"
    chase_interval_seconds: float = 4.0   # wait per quote before re-pegging
    max_chases: int = 5                   # re-quotes before market fallback
    fill_poll_seconds: float = 0.5        # order-status polling cadence

    # System.
    db_path: str = "data/journal.sqlite"
    poll_grace_seconds: int = 10         # wait after bar close before fetching
    binance_base: str = "https://fapi.binance.com"
    telegram_token: str = ""
    telegram_chat_id: str = ""

    def mode_label(self) -> str:
        """Journal tag for this environment: "testnet" or "live"."""
        return "testnet" if "testnet" in self.binance_base else "live"

    def __post_init__(self) -> None:
        if self.order_style not in ("bbo", "market"):
            raise ValueError(
                f"order_style must be bbo|market, got {self.order_style!r}")
        if self.z_out >= self.z_in:
            raise ValueError("z_out must be < z_in")
        if self.win_mu <= 0 or self.win_sd <= 0:
            raise ValueError("windows must be positive")


def load_config(path: Optional[str] = None) -> Config:
    """Builds a Config from an optional JSON file plus environment overrides.

    Environment variables TWOPAIR_TELEGRAM_TOKEN / TWOPAIR_TELEGRAM_CHAT_ID
    override file values so secrets stay out of committed JSON.

    Args:
        path: JSON file with a subset of Config fields, or None for defaults.

    Returns:
        A validated Config.
    """
    data: dict[str, Any] = {}
    if path is not None:
        data = json.loads(pathlib.Path(path).read_text())
        unknown = set(data) - {f.name for f in dataclasses.fields(Config)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for env, key in (("TWOPAIR_TELEGRAM_TOKEN", "telegram_token"),
                     ("TWOPAIR_TELEGRAM_CHAT_ID", "telegram_chat_id")):
        if os.environ.get(env):
            data[key] = os.environ[env]
    return Config(**data)
