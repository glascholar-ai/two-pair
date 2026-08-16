#!/usr/bin/env python3
"""Q4 (Binance-only part): perp price behaviour around dividend 'Special' funding.

For every Special funding event (ex-date 00:00 UTC, rate = -D/M): perp log return over
23:55 -> 00:05 (the settlement bar), 00:00 -> 08:00 (DEAD), 08:00 -> 13:30 (PRE), and the
13:25 -> 13:35 open bar, all in bps, compared with the dividend size and with the same
symbol's unconditional distribution of those windows.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd

from scan_basis_prep import load_bn, load_special_funding, minute_of_day


def _px_at(lp: pd.Series, t: pd.Timestamp) -> float:
    try:
        return float(cast(Any, lp.at[t]))
    except KeyError:
        return np.nan


def window_ret(lp: pd.Series, t0: pd.Timestamp, m0: int, m1: int) -> float:
    a = _px_at(lp, cast(pd.Timestamp, t0 + pd.Timedelta(minutes=m0)))
    b = _px_at(lp, cast(pd.Timestamp, t0 + pd.Timedelta(minutes=m1)))
    return (b - a) * 1e4


def uncond(lp: pd.Series, m0: int, m1: int) -> Dict[str, float]:
    """Unconditional distribution of the same window across all weekdays."""
    idx = pd.DatetimeIndex(lp.index)
    mod = minute_of_day(idx)
    days = pd.DatetimeIndex((idx - pd.to_timedelta(mod, unit="m")).unique())
    vals = [window_ret(lp, cast(pd.Timestamp, d), m0, m1) for d in days if d.dayofweek < 5]
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()), "std": float(v.std()), "n": float(len(v))}


def main(argv: List[str]) -> None:
    sp = load_special_funding()
    if argv[1:]:
        sp = cast(pd.DataFrame, sp[sp["sym"].isin(argv[1:])])
    rows: List[Dict[str, object]] = []
    for sym, t, rate in zip(sp["sym"], sp["t"], sp["rate"]):
        sym = str(sym)
        try:
            px = load_bn(sym)["c"]
        except FileNotFoundError:
            continue
        lp = pd.Series(np.log(np.asarray(px.values, dtype=float)), index=px.index)
        t0 = cast(pd.Timestamp, cast(pd.Timestamp, pd.Timestamp(cast(Any, t))).normalize())
        div = -float(rate) * 1e4
        u = uncond(lp, -5, 5)
        r_settle = window_ret(lp, t0, -5, 5)
        rows.append({"sym": sym, "ex_date": t0.date(), "div_bps": div,
                     "r_2355_0005": r_settle, "z_settle": (r_settle - u["mean"]) / u["std"],
                     "uncond_std_settle": u["std"],
                     "r_AH_prev(20:00->23:55)": window_ret(lp, t0, -240, -5),
                     "r_DEAD(00:05->08:00)": window_ret(lp, t0, 5, 480),
                     "r_PRE(08:00->13:25)": window_ret(lp, t0, 480, 805),
                     "r_open(13:25->13:35)": window_ret(lp, t0, 805, 815),
                     "r_-24h(prev 00:00->23:55)": window_ret(lp, t0, -1440, -5)})
    df = pd.DataFrame(rows)
    print("## Q4 补充: 所有 Special funding 事件的 perp 自身走势 (bps)\n")
    print(str(df.round(1).to_markdown(index=False)) + "\n")
    num = df.drop(columns=["sym", "ex_date"])
    big = cast(pd.DataFrame, num[num["div_bps"] >= 5])
    summ = pd.DataFrame({"all_mean": num.mean(), "all_median": num.median(),
                         "div>=5_mean": big.mean(), "div>=5_median": big.median()}).T
    print("汇总:\n\n" + str(summ.round(1).to_markdown()) + "\n")
    x = np.asarray(big["div_bps"], dtype=float)
    y = np.asarray(big["r_2355_0005"], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 3:
        X = np.column_stack([np.ones(int(ok.sum())), x[ok]])
        beta = np.linalg.lstsq(X, y[ok], rcond=None)[0]
        resid = y[ok] - X @ beta
        se = float(np.sqrt((resid @ resid) / (ok.sum() - 2) * np.linalg.inv(X.T @ X)[1, 1]))
        print(f"回归 r_2355_0005 = {beta[0]:.1f} + {beta[1]:.2f} × div (SE {se:.2f}, n={int(ok.sum())}); "
              f"若 perp 在结算时刻同步除息, 斜率应≈-1\n")
        for col in ("r_DEAD(00:05->08:00)", "r_PRE(08:00->13:25)", "r_open(13:25->13:35)"):
            yy = np.asarray(big[col], dtype=float)
            okk = np.isfinite(x) & np.isfinite(yy)
            b2 = np.linalg.lstsq(np.column_stack([np.ones(int(okk.sum())), x[okk]]), yy[okk], rcond=None)[0]
            print(f"回归 {col} ~ div: 斜率 {b2[1]:.2f}\n")


if __name__ == "__main__":
    main(sys.argv)
