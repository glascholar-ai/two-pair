#!/usr/bin/env python3
"""Refreshes the pair dataset and funding CSVs used by the backtest.

Usage: python3 scripts/refresh_data.py [--start 2026-07-08]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import cast

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from twopair import data as datamod
from twopair.config import load_config


def main() -> None:
    """Fetches klines, FX, and funding, then writes the data/ CSVs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-07-08",
                        help="UTC start date for history")
    args = parser.parse_args()
    cfg = load_config()
    raw = pd.Timestamp(args.start, tz="UTC")
    if bool(pd.isna(cast(object, raw))):
        raise SystemExit(f"invalid --start date: {args.start!r}")
    start_ms = int(cast(pd.Timestamp, raw).timestamp() * 1000)

    kr = datamod.fetch_klines(cfg.kr_symbol, "5m", start_ms)
    us = datamod.fetch_klines(cfg.us_symbol, "5m", start_ms)
    fx = datamod.fetch_fx_yahoo()
    pair = datamod.build_pair_dataset(kr, us, fx)
    pair.index.name = "ts"
    pair.to_csv("data/skhx_pair_5m.csv")
    print(f"pair: {len(pair)} bars {pair.index.min()} -> {pair.index.max()}")

    for symbol, path in ((cfg.kr_symbol, "data/funding_kr.csv"),
                         (cfg.us_symbol, "data/funding_us.csv")):
        ser = datamod.fetch_funding(symbol, start_ms)
        datamod.save_funding_csv(ser, path)
        print(f"funding {symbol}: {len(ser)} settlements -> {path}")


if __name__ == "__main__":
    main()
