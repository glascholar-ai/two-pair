#!/usr/bin/env python3
"""Apply main-pair ideas to the new pairs: session diagnostics + variants.

Variants vs the research baseline (plain 10d z, taker 20bps, funding):
  A. plain sd (research baseline replication)
  B. session-segmented sd (main-pair mechanism)
  C. A + MTM stop overlay (-250 bps)
  D. B + MTM stop overlay
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sps

D = Path(__file__).parent / "data" / "bn5m"
WIN = 2880          # 10d anchor
SD_SEG_WIN = 600    # trailing same-segment bars (~10 sessions of depth)
Z_IN, Z_EXIT, Z_STOP = 2.0, 0.5, 4.0
MAX_HOLD = 288 * 14
COST = 20.0
MTM_STOP_BPS = 250.0


def seg_of(ts: pd.Timestamp) -> str:
    if ts.weekday() >= 5:
        return "wknd"
    hm = ts.hour * 60 + ts.minute
    if hm < 390:
        return "KR_open"
    if hm < 810:
        return "KR->US"
    if hm < 1200:
        return "US_open"
    return "US->KR"


def load(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(D / f"{sym}USDT.parquet")
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("dt")[["o", "c"]]


def load_funding() -> dict:
    f = pd.read_parquet(D / "_funding.parquet")
    f["fundingRate"] = f["fundingRate"].astype(float)
    f["dt"] = pd.to_datetime(f["fundingTime"].astype("int64"), unit="ms",
                             utc=True)
    f["sym"] = f["symbol"].str[:-4]
    return {s: g.set_index("dt")["fundingRate"].sort_index()
            for s, g in f.groupby("sym")}


def diagnostics(a: str, b: str, df: pd.DataFrame) -> None:
    lr = np.log(df["c_a"] / df["c_b"])
    dlr = (lr.diff() * 1e4).dropna()
    segs = pd.Series([seg_of(t) for t in dlr.index], index=dlr.index)
    print(f"--- {a}/{b}: 5m ratio-change vol by segment (bps) ---")
    groups = []
    for s in ("KR_open", "KR->US", "US_open", "US->KR", "wknd"):
        x = dlr[segs == s]
        groups.append(x.values)
        print(f"  {s:8s} n={len(x):5d}  std={x.std():6.2f}  "
              f"lag1_ac={x.autocorr(1):+.3f}")
    w, p = sps.levene(*groups[:4], center="median")
    print(f"  Brown-Forsythe (4 weekday segs): W={w:.1f} p={p:.2e}  "
          f"vol ratio max/min={max(g.std() for g in groups)/min(g.std() for g in groups):.2f}")


def backtest(df: pd.DataFrame, fa, fb, segmented: bool,
             mtm_stop: bool) -> pd.DataFrame:
    lr = np.log(df["c_a"] / df["c_b"])
    mu = lr.rolling(WIN).mean()
    resid = lr - mu
    if segmented:
        segs = pd.Series([seg_of(t) for t in df.index], index=df.index)
        sd = resid.groupby(segs).transform(
            lambda x: x.rolling(SD_SEG_WIN, min_periods=200).std())
    else:
        sd = lr.rolling(WIN).std()
    z = (resid / sd).to_numpy()
    o_a, o_b = df["o_a"].to_numpy(), df["o_b"].to_numpy()
    idx = df.index
    trades, pos, n = [], 0, len(df)
    i0 = 0
    for i in range(WIN, n - 1):
        if np.isnan(z[i]):
            continue
        if pos == 0:
            if z[i] > Z_IN:
                pos, i0 = -1, i + 1
            elif z[i] < -Z_IN:
                pos, i0 = 1, i + 1
        else:
            mtm = pos * ((np.log(o_a[i] / o_a[i0])
                          - np.log(o_b[i] / o_b[i0])) * 1e4)
            stop_z = abs(z[i]) > Z_STOP and np.sign(z[i]) == -pos
            stop_m = mtm_stop and mtm <= -MTM_STOP_BPS
            if (abs(z[i]) < Z_EXIT or stop_z or stop_m
                    or (i - i0) >= MAX_HOLD or i == n - 2):
                i1 = i + 1
                gross = pos * ((np.log(o_a[i1] / o_a[i0])
                                - np.log(o_b[i1] / o_b[i0])) * 1e4)
                fw_a = fa.loc[idx[i0]:idx[i1]].sum() if len(fa) else 0.0
                fw_b = fb.loc[idx[i0]:idx[i1]].sum() if len(fb) else 0.0
                fcost = (pos * fw_a - pos * fw_b) * 1e4
                trades.append({"net": gross - fcost - COST,
                               "stopped_z": stop_z, "stopped_m": stop_m})
                pos = 0
    return pd.DataFrame(trades)


def main() -> None:
    fund = load_funding()
    for a, b in (("EWY", "SAMSUNG"), ("MU", "DRAM")):
        pa, pb = load(a), load(b)
        df = pa.join(pb, lsuffix="_a", rsuffix="_b", how="inner").dropna()
        diagnostics(a, b, df)
        print(f"--- {a}/{b}: variant comparison ---")
        for name, seg, mtm in (("A plain (baseline)", False, False),
                               ("B segmented sd", True, False),
                               ("C plain + MTM stop", False, True),
                               ("D segmented + MTM", True, True)):
            t = backtest(df, fund.get(a, pd.Series(dtype=float)),
                         fund.get(b, pd.Series(dtype=float)), seg, mtm)
            if not len(t):
                print(f"  {name:22s} no trades")
                continue
            print(f"  {name:22s} n={len(t):3d}  win={((t.net > 0).mean() * 100):3.0f}%  "
                  f"med={t.net.median():+7.1f}  total={t.net.sum():+8.0f}  "
                  f"worst={t.net.min():+7.0f}  zstop={t.stopped_z.sum()}  "
                  f"mstop={t.stopped_m.sum()}")
        print()


if __name__ == "__main__":
    main()
