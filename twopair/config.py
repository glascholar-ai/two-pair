"""Strategy and system configuration.

A single frozen dataclass carries every tunable. Defaults reproduce the
baseline v3 backtest exactly; live-only fields (symbols, notional, paths,
credentials) sit alongside so one object configures the whole system.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
from typing import Any, Optional


@dataclasses.dataclass(frozen=True)
class Config:
    """All strategy, risk, and system parameters."""

    # Pair label (journal tag prefix, notifications, data file names).
    pair_name: str = "skhx"

    # Instruments. Historical field names: kr_symbol = leg A, us_symbol =
    # leg B (any two symbols; the KR/US naming is from the first pair).
    kr_symbol: str = "SKHYNIXUSDT"
    us_symbol: str = "SKHYUSDT"

    # Signal parameters (baseline v3 — see README.md before changing).
    # For the main pair win_mu must stay at 24h multiples (intraday
    # seasonality); slow pairs use e.g. 2880 (10d) with segmented_sd=False.
    win_mu: int = 288          # anchor window, 5m bars
    min_mu: int = 144
    win_sd: int = 300          # sd window (per segment when segmented)
    min_sd: int = 100
    segmented_sd: bool = True  # session-conditional sd (main-pair mechanism;
                               # actively harmful on slow pairs — validated)
    fx_source: str = "usdkrw"  # "usdkrw" (lr subtracts ln FX) | "none" (fx=1)
    z_in: float = 2.0
    z_out: float = 0.5
    max_hold_hours: float = 24.0
    mtm_stop_pct: float = 2.5           # 0 disables
    z_stop: float = 0.0                 # same-direction |z| stop; 0 disables
    mtm_take_profit_pct: float = 0.0    # close when MTM >= this; 0 disables
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
    deadman_seconds: int = 900            # exchange auto-cancels resting
                                          # orders this long after the last
                                          # heartbeat; 0 disables

    # System.
    digest_utc_hour: int = 0             # daily Telegram digest after this
                                         # UTC hour; -1 disables
    db_path: str = "data/journal.sqlite"
    poll_grace_seconds: int = 10         # wait after bar close before fetching
    binance_base: str = "https://fapi.binance.com"
    portfolio_margin: bool = False       # account uses Binance Portfolio
                                         # Margin: signed calls go to papi
                                         # /papi/v1/um/* endpoints
    telegram_token: str = ""
    telegram_chat_id: str = ""

    def mode_label(self) -> str:
        """Journal tag for this environment: "testnet" or "live"."""
        return "testnet" if "testnet" in self.binance_base else "live"

    def warmup_days(self) -> int:
        """History needed to fill the signal windows before going live."""
        bars_needed = max(self.win_mu, self.win_sd) * 1.5
        return max(7, math.ceil(bars_needed / (86400 / self.bar_seconds)))

    def __post_init__(self) -> None:
        if self.order_style not in ("bbo", "market"):
            raise ValueError(
                f"order_style must be bbo|market, got {self.order_style!r}")
        if self.fx_source not in ("usdkrw", "none"):
            raise ValueError(
                f"fx_source must be usdkrw|none, got {self.fx_source!r}")
        if self.z_out >= self.z_in:
            raise ValueError("z_out must be < z_in")
        if self.z_stop and self.z_stop <= self.z_in:
            raise ValueError("z_stop must exceed z_in (or be 0)")
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
