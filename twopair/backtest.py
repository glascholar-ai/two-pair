"""Backtest runner built on the shared SignalEngine + Strategy.

Because the exact same classes drive live trading, this module doubles as
the parity reference: any change to the signal path shows up here first.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
from typing import List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from twopair.config import Config
from twopair.data import funding_between
from twopair.signal import Bar, SignalEngine
from twopair.strategy import Strategy, Trade


@dataclasses.dataclass(frozen=True)
class BacktestResult:
    """Trades plus the bar-level mark-to-market equity curve and metrics."""

    trades: List[Trade]
    equity: pd.Series          # % of single-leg notional, indexed by bar ts
    total_pct: float
    annualized_pct: float
    sharpe: float
    max_drawdown_pct: float
    in_market_frac: float

    def trades_frame(self) -> pd.DataFrame:
        """Trades as a DataFrame for inspection or CSV export."""
        rows = [dataclasses.asdict(t) for t in self.trades]
        for row in rows:
            row["reason"] = row["reason"].value
        return pd.DataFrame(rows)


def run_backtest(pair: pd.DataFrame, funding_kr: pd.Series,
                 funding_us: pd.Series, cfg: Config) -> BacktestResult:
    """Replays a pair dataset through the shared signal/strategy stack.

    Args:
        pair: DataFrame with columns kr, us, fx indexed by UTC bar time.
        funding_kr: KR-leg funding settlements (fraction), UTC indexed.
        funding_us: US-leg funding settlements (fraction), UTC indexed.
        cfg: Strategy configuration.

    Returns:
        BacktestResult with trades, equity curve, and summary metrics.
    """
    engine = SignalEngine(cfg.win_mu, cfg.min_mu, cfg.win_sd, cfg.min_sd)
    strat = Strategy(cfg)

    equity: List[float] = []
    eq = 0.0
    prev: Optional[Tuple[dt.datetime, float]] = None
    kr_col = pair["kr"].to_numpy(dtype=float)
    us_col = pair["us"].to_numpy(dtype=float)
    fx_col = pair["fx"].to_numpy(dtype=float)
    for i, raw_ts in enumerate(pair.index):
        ts = cast(dt.datetime, raw_ts)
        sig = engine.update(Bar(ts=ts, kr=kr_col[i], us=us_col[i],
                                fx=fx_col[i]))
        dlr = 0.0
        funding_pct = 0.0
        if strat.position is not None and prev is not None:
            dlr = sig.lr - prev[1]
            kr_f = funding_between(funding_kr, prev[0], ts)
            us_f = funding_between(funding_us, prev[0], ts)
            funding_pct = (-kr_f + us_f) * 100.0
            eq += strat.position.side * (dlr * 100.0 + funding_pct)
        strat.on_bar(sig, dlr, funding_pct)
        equity.append(eq)
        prev = (ts, sig.lr)

    curve = pd.Series(equity, index=pair.index, name="equity_pct")
    return BacktestResult(trades=list(strat.trades), equity=curve,
                          **_metrics(curve))


def _metrics(curve: pd.Series) -> dict:
    """Summary metrics from a bar-level equity curve."""
    drawdown = curve - curve.cummax()
    daily = curve.resample("1D").last().dropna().diff().dropna()
    span = (pd.Timestamp(str(curve.index[-1]))
            - pd.Timestamp(str(curve.index[0])))
    n_days = span.total_seconds() / 86400.0
    total = float(curve.iloc[-1])
    ann = total / n_days * 365.0 if n_days > 0 else math.nan
    vol = float(daily.std())
    sharpe = float(daily.mean()) / vol * math.sqrt(365.0) if vol > 0 else math.nan
    moved = curve.diff().fillna(0.0) != 0
    return {
        "total_pct": total,
        "annualized_pct": ann,
        "sharpe": sharpe,
        "max_drawdown_pct": float(drawdown.min()),
        "in_market_frac": float(np.mean(moved)),
    }
