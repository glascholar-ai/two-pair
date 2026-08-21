#!/usr/bin/env python3
"""Hindsight-optimal capital allocation upper bound for the type-A book.

Two steps:
1. Enumerate the full opportunity set by re-running run_type_a with unlimited
   capital (entry/exit signals are capital-independent, so each name's trade
   sequence is invariant; per-trade max size u_i = min(15% OI slot, per-name
   cap) is recorded as the sim's notional).
2. LP:  max sum(r_i * x_i)  s.t.  0 <= x_i <= u_i  and, at every entry time t,
   sum over trades whose [t_in, t_out) covers t of x_i <= CAPITAL.
   Divisible notionals -> plain LP (the 0/1 variant would be NP-hard; we want
   the evaluation bound, so fractional is the right relaxation anyway).

Outputs the bound, the greedy-realized figure for comparison, and where the
oracle concentrates.
"""
from __future__ import annotations

import json
import time
from typing import cast

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

import scan_dyn_backtest as bt

CAPITAL = 5_000_000.0


def main() -> None:
    t1 = (int(time.time() * 1000) // 86_400_000) * 86_400_000
    t0 = t1 - bt.WINDOW_DAYS * 86_400_000
    cand = json.loads((bt.DYN / "candidates.json").read_text())
    ib_map = json.loads((bt.DYN / "ib_map.json").read_text())
    ib_ok = {r["ticker"]: r for r in ib_map if r.get("status") == "ok"}
    fb = bt.FundingBook()

    # step 1: unlimited capital -> full opportunity set
    bt.CAPITAL_STOCK = bt.CAPITAL_PERP = 1e12
    closed, _ = bt.run_type_a(cand["type_a"], ib_ok, fb, t0, t1)
    tr = pd.DataFrame(closed)
    tr = cast(pd.DataFrame, tr[tr["notional"] > 0]).reset_index(drop=True)
    n = len(tr)
    r = tr["net_bps"].to_numpy(dtype=float) / 1e4
    u = tr["notional"].to_numpy(dtype=float)
    tin = tr["t_in"].to_numpy(dtype="int64")
    tout = tr["t_out"].to_numpy(dtype="int64")
    print(f"opportunity set: {n} trades, gross max notional "
          f"sum={u.sum()/1e6:.1f}M, {int((r > 0).sum())} positive")

    # step 2: LP. Concurrency constraints at each distinct entry time.
    checkpoints = np.unique(tin)
    A = lil_matrix((len(checkpoints), n))
    for row, t in enumerate(checkpoints):
        active = (tin <= t) & (tout > t)
        A[row, np.where(active)[0]] = 1.0
    res = linprog(c=-r, A_ub=A.tocsr(), b_ub=np.full(len(checkpoints), CAPITAL),
                  bounds=list(zip(np.zeros(n), u)), method="highs")
    if not res.success:
        raise SystemExit(f"LP failed: {res.message}")
    x = cast(np.ndarray, res.x)
    pnl = float(r @ x)
    print(f"\nORACLE bound: ${pnl:,.0f} over {bt.WINDOW_DAYS}d "
          f"(ann {pnl/(bt.WINDOW_DAYS/365)/1e7*100:.1f}% on $10M)")
    used = pd.DataFrame({"key": tr["key"], "mode": tr["mode"],
                         "t_in": tr["t_in"], "days": tr["days"],
                         "net_bps": tr["net_bps"], "u": u, "x": x,
                         "pnl": r * x})
    top = cast(pd.DataFrame, used[used["x"] > 1e3].groupby("key").agg(
        n=("x", "size"), alloc_avg=("x", "mean"),
        days=("days", "sum"), pnl=("pnl", "sum")).round(0))
    top = top.sort_values("pnl", ascending=False)
    print(f"\noracle allocation ({int((x > 1e3).sum())} of {n} trades funded, "
          f"fully-funded share {float((x >= u - 1e3).mean()):.0%}):")
    print(top.head(20).to_string())
    loads = np.asarray(A.tocsr() @ x).ravel()
    n_bind = int((loads >= CAPITAL - 1e3).sum())
    print(f"\ncapital binding at {n_bind}/{len(checkpoints)} checkpoints, "
          f"median concurrent demand {np.median(loads)/1e6:.1f}M")


if __name__ == "__main__":
    main()
