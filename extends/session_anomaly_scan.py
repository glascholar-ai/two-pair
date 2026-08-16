#!/usr/bin/env python3
"""Quick scan of 24h-specific anomalies in Binance US-equity perps.

Segments each US trading day (UTC, EDT session 13:30-20:00) into:
  AH   20:00-00:00   (US after-hours, real AH market exists)
  DEAD 00:00-08:00   (no US venue open; perp is the only price)
  PRE  08:00-13:30   (US pre-market)
  OPEN 13:30-14:30   (first hour of RTH)
  REST 14:30-20:00
Tests: reversal/continuation of off-hours moves at open, extreme off-hours
5m moves, weekend drift, BTC beta by segment, funding by slot.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Dict, List, cast

import numpy as np
import pandas as pd

D = Path(__file__).parent / "data" / "bn5m"
NON_US = {"SAMSUNG", "SKHYNIX", "HYUNDAI", "CSOPSAMSUNG2L", "CSOPSKHYNIX2L", "SONY",
          "HK0700", "HK1810", "TENCENT", "MEITUAN", "KUAISHOU", "POPMART", "GIGADEV",
          "MINIMAX", "ZHIPU", "OPENAI", "ANTHROPIC", "SPCX", "SPCXUSD1", "CBRS", "BNC",
          "BOT", "SHAZ", "KSTR", "FWDI", "STRC", "BSP", "AXTI", "QNTX", "PENG", "BBX",
          "USAR", "MUU", "MVLL", "INTW", "SNXX", "TQQQ", "SQQQ", "SOXL", "SOXS", "TZA",
          "TMF", "TBT", "UVXY", "KORU", "BITO", "SKHY"}
SEG = [("AH", 20 * 60, 24 * 60), ("DEAD", 0, 8 * 60), ("PRE", 8 * 60, 13 * 60 + 30),
       ("OPEN", 13 * 60 + 30, 14 * 60 + 30), ("REST", 14 * 60 + 30, 20 * 60)]


def lg(s: pd.Series) -> pd.Series:
    return cast(pd.Series, np.log(s))


def didx(s: pd.Series) -> pd.DatetimeIndex:
    return cast(pd.DatetimeIndex, s.index)


def minute_of_day(idx: pd.DatetimeIndex) -> np.ndarray:
    return (np.asarray(idx.asi8, dtype=np.int64) // 60_000_000_000) % 1440


def dow(idx: pd.DatetimeIndex) -> np.ndarray:
    return (np.asarray(idx.asi8, dtype=np.int64) // 86_400_000_000_000 + 3) % 7


def normalize(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return cast(pd.DatetimeIndex, idx - pd.to_timedelta(minute_of_day(idx), unit="m"))


def load(sym: str) -> pd.Series:
    df = pd.read_parquet(D / f"{sym}USDT.parquet")
    s = pd.Series(df["c"].values, index=pd.to_datetime(df["ts"], unit="ms", utc=True))
    return cast(pd.Series, s[~s.index.duplicated()]).sort_index()


def seg_returns(px: pd.Series) -> pd.DataFrame:
    """Per US-trading-day segment log returns; AH/DEAD/PRE are the ones preceding open."""
    lp = lg(px)
    idx = didx(lp)
    minute = minute_of_day(idx)
    day = pd.Series(normalize(idx), index=idx)
    # AH belongs to the *next* trading day
    key = day.where(minute < 20 * 60, day + pd.Timedelta(days=1))
    rows: Dict[pd.Timestamp, Dict[str, float]] = {}
    for name, a, b in SEG:
        m = (minute >= a) & (minute < b)
        g = cast(pd.Series, lp[m]).groupby(key[m])
        r = cast(pd.Series, g.last() - g.first())
        for d, v in r.items():
            rows.setdefault(cast(pd.Timestamp, d), {})[name] = float(v)
    out = pd.DataFrame(rows).T.sort_index()
    out = out[dow(cast(pd.DatetimeIndex, out.index)) < 5].dropna()
    return cast(pd.DataFrame, out)


def main() -> None:
    syms = [p.stem[:-4] for p in D.glob("*USDT.parquet") if p.stem[:-4] not in NON_US]
    allseg: List[pd.DataFrame] = []
    px_all: Dict[str, pd.Series] = {}
    for s in syms:
        px = load(s)
        if len(px) < 30 * 288:
            continue
        px_all[s] = px
        sg = seg_returns(px)
        sg["sym"] = s
        allseg.append(sg)
    seg = pd.concat(allseg)
    print(f"symbols: {len(px_all)}, symbol-days: {len(seg)}")
    seg["OFF"] = seg["AH"] + seg["DEAD"] + seg["PRE"]
    seg["RTH"] = seg["OPEN"] + seg["REST"]

    print("\n== 1. mean/std of segment log-returns (bps) ==")
    print((seg[["AH", "DEAD", "PRE", "OPEN", "REST", "OFF", "RTH"]] * 1e4)
          .agg(["mean", "std"]).T.round(1))

    print("\n== 2. does an off-hours move continue or revert? (pooled OLS slope, t) ==")
    for x, y in [("DEAD", "OPEN"), ("DEAD", "RTH"), ("OFF", "OPEN"), ("OFF", "RTH"),
                 ("AH", "DEAD"), ("AH", "PRE"), ("PRE", "OPEN"), ("OPEN", "REST")]:
        xs, ys = seg[x], seg[y]
        b = (xs * ys).sum() / (xs ** 2).sum()
        res = ys - b * xs
        t = b / np.sqrt((res ** 2).sum() / (len(xs) - 1) / (xs ** 2).sum())
        print(f"  {y} ~ {x}: slope {b:+.3f}  t={t:+.1f}")

    print("\n== 3. conditional: big DEAD-zone move (|r|>1.5%) -> OPEN / RTH ==")
    for lo, hi in [(0.015, 0.03), (0.03, 9)]:
        m = seg["DEAD"].abs().between(lo, hi)
        sgn = np.sign(seg.loc[m, "DEAD"])
        print(f"  n={m.sum():4d}  OPEN same-dir mean {(sgn*seg.loc[m,'OPEN']).mean()*1e4:+.0f} bps"
              f"  RTH same-dir mean {(sgn*seg.loc[m,'RTH']).mean()*1e4:+.0f} bps"
              f"  hit-rate(RTH continues) {(sgn*seg.loc[m,'RTH']>0).mean():.2f}")

    print("\n== 4. weekend: Fri 20:00 -> Mon 13:30 vs Monday RTH ==")
    mon = seg[dow(cast(pd.DatetimeIndex, seg.index)) == 0]
    b = (mon["OFF"] * mon["RTH"]).sum() / (mon["OFF"] ** 2).sum()
    print(f"  n={len(mon)}  weekend mean {mon['OFF'].mean()*1e4:+.0f} bps std {mon['OFF'].std()*1e4:.0f}"
          f"  MonRTH~weekend slope {b:+.3f}")

    print("\n== 5. extreme 5m moves by segment: forward 30m/60m return (same-dir, bps) ==")
    recs = []
    for s, px in px_all.items():
        lp = lg(px)
        idx = didx(px)
        r = lp.diff()
        vol = r.rolling(288 * 3).std()
        z = r / vol
        fwd6 = lp.shift(-6) - lp
        fwd12 = lp.shift(-12) - lp
        minute = minute_of_day(idx)
        segname = pd.Series("REST", index=idx)
        for name, a, b in SEG:
            segname[(minute >= a) & (minute < b)] = name
        m = (z.abs() > 4) & (dow(idx) < 5)
        recs.append(pd.DataFrame({"seg": segname[m], "sd6": (np.sign(r) * fwd6)[m],
                                  "sd12": (np.sign(r) * fwd12)[m], "z": z[m]}))
    ex = pd.concat(recs).dropna()
    print(ex.groupby("seg").agg(n=("z", "size"), fwd30m=("sd6", lambda v: v.mean() * 1e4),
                                fwd60m=("sd12", lambda v: v.mean() * 1e4),
                                revert_rate=("sd12", lambda v: (v < 0).mean())).round(2))

    print("\n== 6. BTC beta by segment (pooled, 5m returns) ==")
    url = ("https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m&limit=1500"
           "&endTime={}")
    btc: List[pd.Series] = []
    end = int(pd.Timestamp.utcnow().timestamp() * 1000)
    for _ in range(25):
        try:
            k = json.load(urllib.request.urlopen(url.format(end), timeout=15))
        except Exception as ex_:  # noqa: BLE001
            print("  btc fetch failed:", ex_)
            k = []
        if not k:
            break
        btc.append(pd.Series({int(x[0]): float(x[4]) for x in k}))
        end = k[0][0] - 1
    if btc:
        b = pd.concat(btc).sort_index()
        b.index = pd.to_datetime(b.index, unit="ms", utc=True)
        rb = lg(b).diff()
        rows = []
        for s, px in px_all.items():
            r = lg(px).diff().reindex(rb.index).dropna()
            x = rb.loc[r.index]
            ridx = didx(r)
            minute = minute_of_day(ridx)
            for name, a, bb in SEG:
                m = (minute >= a) & (minute < bb) & (dow(ridx) < 5)
                if m.sum() < 500:
                    continue
                beta = (x[m] * r[m]).sum() / (x[m] ** 2).sum()
                corr = x[m].corr(r[m])
                rows.append({"seg": name, "beta": beta, "corr": corr})
        print(pd.DataFrame(rows).groupby("seg")[["beta", "corr"]].median().round(3))

    print("\n== 7. funding by settlement slot (US equity perps, bps per 8h) ==")
    f = pd.read_parquet(D / "_funding.parquet")
    f = f[f["symbol"].str[:-4].isin(px_all.keys())].copy()
    f["rate"] = f["fundingRate"].astype(float) * 1e4
    f["hour"] = minute_of_day(pd.DatetimeIndex(pd.to_datetime(f["fundingTime"], unit="ms", utc=True))) // 60
    print(f.groupby("hour")["rate"].agg(["mean", "median", "std", "size"]).round(2))
    print("  mean |rate| by hour:", f.groupby("hour")["rate"].apply(lambda v: v.abs().mean()).round(2).to_dict())


if __name__ == "__main__":
    main()
