#!/usr/bin/env python3
"""Drill-down on the resting-limit dead-zone simulation (see scan_offhours_reversal.py).

Per-symbol dispersion, monthly $ PnL, hour-of-day, fee/halflife/k sensitivity and
position concurrency for the DEAD / WKND / AH segments.
Run: python scan_offhours_sim_detail.py [--src bn|hl] [--k 4] [--hold 36]
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple, cast

import pandas as pd

from scan_offhours_reversal import col, load_ohlc, sim_limits, trade_stats, universe

NOTIONAL = 20_000.0


def load_all(src: str, min_days: int) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for s in universe(src):
        df = load_ohlc(src, s)
        if df is not None and len(df) >= min_days * 288:
            out[s] = df
    return out


def run(data: Dict[str, pd.DataFrame], k: float, hold: int, halflife: int = 6,
        segs: Tuple[str, ...] = ("DEAD", "WKND", "AH")) -> pd.DataFrame:
    trs = [sim_limits(df, s, k, hold, halflife=halflife, segs=segs) for s, df in data.items()]
    return pd.concat([t for t in trs if len(t)], ignore_index=True)


def sub(tr: pd.DataFrame, seg: str) -> pd.DataFrame:
    return cast(pd.DataFrame, tr[tr["seg"] == seg])


def per_symbol(tr: pd.DataFrame, fee: float) -> pd.DataFrame:
    tr = tr.assign(net=col(tr, "gross") - fee)
    g = tr.groupby("sym")
    out = pd.DataFrame({"fills": g.size(), "net_bps": g["net"].mean(),
                        "win%": g["net"].apply(lambda v: (v > 0).mean() * 100),
                        "sum_$": cast(pd.Series, g["net"].sum()) * NOTIONAL / 1e4,
                        "worst_bps": g["net"].min()})
    return out.sort_values("fills", ascending=False)


def concurrency(tr: pd.DataFrame) -> pd.Series:
    """Max / p95 simultaneous open positions (across all names, minute resolution)."""
    ts = col(tr, "t")
    ends = ts + pd.to_timedelta(col(tr, "bars") * 5, unit="m")
    ev = pd.concat([pd.Series(1, index=ts), pd.Series(-1, index=ends)]).sort_index()
    open_ = ev.groupby(level=0).sum().cumsum()
    return pd.Series({"max_open": open_.max(), "p95_open": open_.quantile(0.95),
                      "p50_open": open_.median()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="bn")
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--hold", type=int, default=36)
    ap.add_argument("--min-days", type=int, default=30)
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    data = load_all(a.src, a.min_days)
    tr = run(data, a.k, a.hold)
    print(f"src={a.src} k={a.k} hold={a.hold} names={len(data)} trades={len(tr)}")
    for seg in ("DEAD", "WKND", "AH"):
        e = sub(tr, seg)
        if e.empty:
            continue
        print(f"\n===== {seg} =====")
        print(trade_stats(e).round(1).to_string())
        ps = per_symbol(e, 4.0)
        print(f"symbols: {len(ps)}, positive-net symbols: {(ps['net_bps'] > 0).sum()}, "
              f"top-5 fill share {ps['fills'].head(5).sum() / ps['fills'].sum():.2f}, "
              f"top-5 $ share {col(ps, 'sum_$').nlargest(5).sum() / max(col(ps, 'sum_$').sum(), 1):.2f}")
        print(ps.head(15).round(1).to_string())
        print("concurrency:", concurrency(e).round(1).to_dict())
        net = col(e, "gross") - 4.0
        ts = col(e, "t")
        month = ts.dt.strftime("%Y-%m")
        mo = pd.DataFrame({"fills": net.groupby(month).size(),
                           "net_bps": net.groupby(month).mean(),
                           "$": (net * NOTIONAL / 1e4).groupby(month).sum()}).round(1)
        print("by month:\n" + mo.to_string())
        if seg == "DEAD":
            hr = ts.dt.hour
            hh = pd.DataFrame({"fills": net.groupby(hr).size(), "net_bps": net.groupby(hr).mean(),
                               "win%": net.groupby(hr).apply(lambda v: (v > 0).mean() * 100)})
            print("by hour:\n" + hh.round(1).to_string())
            side = col(e, "side").map({1: "buy-dip", -1: "sell-spike"})
            print("by side:\n" + pd.DataFrame({"fills": net.groupby(side).size(),
                                              "net_bps": net.groupby(side).mean()}).round(1).to_string())
    print("\n===== fee sensitivity (DEAD) =====")
    e = sub(tr, "DEAD")
    ets = col(e, "t")
    for lab, fee in [("maker0/taker4", 4.0), ("maker2/taker4", 6.0), ("taker4/taker4", 8.0),
                     ("maker2/taker5", 7.0)]:
        net = col(e, "gross") - fee
        span = max((ets.max() - ets.min()).days, 1)
        print(f"  {lab:15s} net {net.mean():+.1f} bps  $/day {net.sum() * NOTIONAL / 1e4 / span:+.0f}")
    print("\n===== halflife / k sensitivity (DEAD, net bps @ maker0/taker4) =====")
    rows: List[Dict[str, object]] = []
    for hl in (6, 12, 24):
        for k in (3.0, 4.0, 5.0):
            t2 = run(data, k, a.hold, halflife=hl, segs=("DEAD",))
            st = trade_stats(t2)
            rows.append({"halflife": hl, "k": k, "fills/day": float(st["fills/day"]),
                         "net_bps": float(st["net_bps"]), "$/day": float(st["$/day@20k"]),
                         "sharpe": float(st["daily_sharpe"]), "maxDD": float(st["maxDD_$"])})
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
