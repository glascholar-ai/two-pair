#!/usr/bin/env python3
"""Leveraged-twin decay harvest on perps: short the levered contract, hedge with
beta x the base contract, rebalance daily. PnL = variance drag - funding - costs.
"""
import numpy as np
import pandas as pd
from pathlib import Path

D = Path(__file__).parent / "data" / "bn5m"

TWINS = [  # (levered, base, beta)
    ("MUU", "MU", 2), ("MVLL", "MRVL", 2), ("INTW", "INTC", 2), ("SNXX", "SNDK", 2),
    ("TQQQ", "QQQ", 3), ("SQQQ", "QQQ", -3), ("SOXL", "SMH", 3), ("SOXS", "SMH", -3),
]
TAKER = 4e-4          # per leg notional traded
DAY = 288


def load(sym):
    f = D / f"{sym}USDT.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("dt")["c"]


def load_funding():
    f = pd.read_parquet(D / "_funding.parquet")
    f["fundingRate"] = f["fundingRate"].astype(float)
    f["dt"] = pd.to_datetime(f["fundingTime"].astype("int64"), unit="ms", utc=True)
    f["sym"] = f["symbol"].str[:-4]
    return {s: g.set_index("dt")["fundingRate"].sort_index() for s, g in f.groupby("sym")}


def main():
    fund = load_funding()
    rows = []
    for lev, base, beta in TWINS:
        L, B = load(lev), load(base)
        if L is None or B is None:
            print(f"skip {lev} (missing)")
            continue
        df = pd.concat({"L": L, "B": B}, axis=1).dropna()
        daily = df.resample("1D").last().dropna()
        if len(daily) < 15:
            print(f"skip {lev} (only {len(daily)} days)")
            continue
        rL = daily["L"].pct_change().dropna()
        rB = daily["B"].pct_change().dropna()
        # short 1 unit lev + long beta units base (beta<0 => short base too)
        pnl = -rL + beta * rB                        # per 1 unit lev notional, daily
        rebal_cost = (rL - beta * rB).abs() * 0 + (beta * rB - rL).abs() * 0  # placeholder
        # daily rebalance turnover ~ |beta*rB - rL| on the base leg
        cost = (beta * rB - rL).abs() * TAKER
        fL = fund.get(lev, pd.Series(dtype=float))
        fB = fund.get(base, pd.Series(dtype=float))
        w = (daily.index[1], daily.index[-1])
        f_lev = fL.loc[w[0]:w[1]].sum()              # short lev receives +f
        f_base = fB.loc[w[0]:w[1]].sum()             # long beta*base pays beta*f
        net = pnl - cost
        days = len(net)
        gross_pos = 1 + abs(beta)                    # capital proxy: total gross notional
        ann = lambda s: s.sum() / days * 365 / gross_pos * 100
        rows.append({
            "twin": f"{lev} vs {beta}x{base}", "days": days,
            "drag_ann%": ann(pnl), "cost_ann%": -ann(cost),
            "funding_ann%": (f_lev - beta * f_base) / days * 365 / gross_pos * 100,
            "net_ann%": ann(net) + (f_lev - beta * f_base) / days * 365 / gross_pos * 100,
            "daily_std_bps": net.std() * 1e4,
            "worst_day_bps": net.min() * 1e4,
            "hedge_r2": np.corrcoef(rL, beta * rB)[0, 1] ** 2,
        })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\n(returns are % annualized on GROSS notional 1+|beta|; drag = variance harvest before costs)")


if __name__ == "__main__":
    main()
