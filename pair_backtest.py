#!/usr/bin/env python3
"""Baseline backtest CLI — thin wrapper over the shared twopair library.

The signal path (SignalEngine + Strategy) is byte-identical to live trading;
this script only loads data, runs it, and prints the report.

Data files (refresh with scripts/refresh_data.py):
    data/skhx_pair_5m.csv   — pair dataset (ts, kr, us, fx)
    data/funding_kr.csv     — KR-leg funding (ts, rate); fetched if absent
    data/funding_us.csv     — US-leg funding (ts, rate); fetched if absent
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib

import pandas as pd

from twopair import data as datamod
from twopair.backtest import run_backtest
from twopair.config import load_config


def _funding(symbol: str, path: str, start_ms: int) -> pd.Series:
    """Loads cached funding or fetches and caches it."""
    if pathlib.Path(path).exists():
        return datamod.load_funding_csv(path)
    ser = datamod.fetch_funding(symbol, start_ms)
    datamod.save_funding_csv(ser, path)
    return ser


def main() -> None:
    """Runs the baseline backtest and prints trades plus equity metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="JSON config path")
    parser.add_argument("--mtm-stop", type=float, default=None,
                        help="override MTM stop in %% (0 disables)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mtm_stop is not None:
        cfg = dataclasses.replace(cfg, mtm_stop_pct=args.mtm_stop)

    pair = datamod.load_pair_csv("data/skhx_pair_5m.csv")
    start_ms = int(pair.index[0].timestamp() * 1000)
    funding_kr = _funding(cfg.kr_symbol, "data/funding_kr.csv", start_ms)
    funding_us = _funding(cfg.us_symbol, "data/funding_us.csv", start_ms)

    result = run_backtest(pair, funding_kr, funding_us, cfg)
    frame = result.trades_frame()
    if not frame.empty:
        cols = ["entry_ts", "entry_seg", "entry_z", "max_abs_z", "held_hours",
                "pnl_pct", "reason"]
        print(frame[cols].round(3).to_string(index=False))
        print()
        wins = (frame["pnl_pct"] > 0).mean() * 100
        stops = (frame["reason"] == "stop").sum()
        print(f"n={len(frame)}  mean {frame['pnl_pct'].mean():+.3f}%  "
              f"median {frame['pnl_pct'].median():+.3f}%  win {wins:.0f}%  "
              f"sum {frame['pnl_pct'].sum():+.2f}%  "
              f"worst {frame['pnl_pct'].min():+.2f}%  stops {stops}")
        print(frame.groupby("entry_seg")["pnl_pct"]
              .agg(["count", "mean", "sum"]).round(3).to_string())
    stop_label = cfg.mtm_stop_pct if cfg.mtm_stop_pct > 0 else "off"
    print(f"\nequity (% of single-leg notional; gross exposure 2x in "
          f"position; mtm_stop={stop_label}):")
    print(f"total {result.total_pct:+.2f}%  ann {result.annualized_pct:+.1f}%  "
          f"Sharpe(daily,365) {result.sharpe:.2f}  "
          f"maxDD {result.max_drawdown_pct:.2f}%  "
          f"in-market {result.in_market_frac * 100:.0f}%")
    frame.round(4).to_csv("data/pair_trades_baseline.csv", index=False)
    result.equity.to_frame("equity_pct").to_csv("data/pair_equity_curve.csv")


if __name__ == "__main__":
    main()
