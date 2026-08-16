#!/usr/bin/env python3
"""Item 3: earnings events - perp behaviour after the print when the cash tape is thin/closed.

Earnings dates come from SEC EDGAR: 8-K filings with Item 2.02 (Results of Operations),
using the EDGAR acceptance timestamp (UTC) to classify AMC (>=20:00 UTC, i.e. after the
16:00 ET close) vs BMO (<13:30 UTC).  Foreign private issuers (6-K filers: TSM, ASML,
BABA, ARM, NOK, SONY, NVO...) are not covered.  Cached to data/edgar_8k_202.parquet.

Windows (UTC, AMC events, t0 = 20:00 of the release day):
  r5/r15/r30/r60 : 20:00 -> 20:05/20:15/20:30/21:00
  AH   : 20:00 -> 00:00        DEAD : 00:00 -> 08:00 (perp only)
  PRE  : 08:00 -> 13:30        OPEN : 13:30 -> 14:30       RTH: 13:30 -> 20:00
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Dict, List, Optional, cast

import numpy as np
import pandas as pd

import scan_events_common as C

CACHE = C.ROOT / "data" / "edgar_8k_202.parquet"
UA = "luna research glascholar@gmail.com"
BN_TO_SEC = {"BRKB": "BRK-B", "PAYP": "PYPL"}


def _get(url: str) -> Optional[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except Exception as ex:  # noqa: BLE001
            print(f"  retry {a} {url}: {ex!r}")
            time.sleep(2)
    return None


def fetch_edgar(syms: List[str]) -> pd.DataFrame:
    """8-K item 2.02 filings (date, acceptance UTC) for each symbol; cached."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    tick = _get("https://www.sec.gov/files/company_tickers.json") or {}
    cik = {v["ticker"]: int(v["cik_str"]) for v in tick.values()}
    rows: List[Dict[str, str]] = []
    for s in syms:
        t = BN_TO_SEC.get(s, s)
        if t not in cik:
            print(f"  no CIK for {s}")
            continue
        d = _get(f"https://data.sec.gov/submissions/CIK{cik[t]:010d}.json")
        time.sleep(0.15)
        if not d:
            continue
        f = d["filings"]["recent"]
        for i in range(len(f["form"])):
            if f["form"][i] == "8-K" and "2.02" in (f["items"][i] or ""):
                rows.append({"sym": s, "date": f["filingDate"][i], "acc": f["acceptanceDateTime"][i]})
    out = pd.DataFrame(rows)
    out.to_parquet(CACHE, index=False)
    return out


def classify(ev: pd.DataFrame) -> pd.DataFrame:
    """Keep 2026-04..08 events; AMC if accepted >=20:00 UTC, BMO if <13:30 UTC."""
    ev = ev.copy()
    ev["acc"] = pd.to_datetime(ev["acc"], utc=True)
    acc = C.col(ev, "acc")
    ev = cast(pd.DataFrame, ev[(acc >= C.utc("2026-03-25")) & (acc <= C.utc("2026-08-13"))])
    acc = C.col(ev, "acc")
    m = acc.dt.hour * 60 + acc.dt.minute
    ev["kind"] = np.where(m >= 20 * 60, "AMC", np.where(m < 13 * 60 + 30, "BMO", "MID"))
    # AMC filings after midnight UTC belong to the previous US date - not seen in practice
    ev["t0"] = [C.utc(str(a.date()), "20:00") if k == "AMC" else
                (C.utc(str(a.date()), "12:30") if k == "BMO" and (a.hour * 60 + a.minute) >= 12 * 60
                 else a.floor("5min")) for a, k in zip(acc, C.col(ev, "kind"))]
    return ev.sort_values("t0").reset_index(drop=True)


def amc_rows(ev: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for sym_, g in ev[ev["kind"] == "AMC"].groupby("sym"):
        sym = cast(str, sym_)
        df = C.load_px(sym)
        s = C.load_close(sym)
        f = cast(pd.DataFrame, fund[fund["symbol"] == sym + "USDT"].copy())
        f["t"] = pd.to_datetime(f["fundingTime"], unit="ms", utc=True)
        f["rate"] = f["fundingRate"].astype(float)
        rate, ft = C.col(f, "rate"), C.col(f, "t")
        norm_abs = float(rate.abs().median()) if len(f) else float("nan")
        for _, e in g.iterrows():
            t0 = C.as_ts(e["t0"])
            if t0 - C.days(3) < C.idx_min(s) or t0 + C.days(1) > C.idx_max(s):
                continue
            d = str(t0.date())
            n1 = t0 + C.days(1)
            while n1.dayofweek >= 5 or n1.date() in {h.date() for h in C.US_HOLIDAYS}:
                n1 += C.days(1)
            nd = str(n1.date())
            gap = C.mins(30)
            r: Dict[str, float | str] = {"sym": sym, "date": d, "acc": str(e["acc"])[11:16]}
            for k, mm in [("r5", 5), ("r15", 15), ("r30", 30), ("r60", 60)]:
                r[k] = C.ret(s, t0, t0 + C.mins(mm), gap)
            r["ah"] = C.ret(s, t0, C.utc(nd, "00:00"), gap)
            r["ah_post30"] = C.ret(s, t0 + C.mins(30), C.utc(nd, "00:00"), gap)
            r["dead"] = C.ret(s, C.utc(nd, "00:00"), C.utc(nd, "08:00"), gap)
            r["pre"] = C.ret(s, C.utc(nd, "08:00"), C.utc(nd, "13:30"), gap)
            r["open1h"] = C.ret(s, C.utc(nd, "13:30"), C.utc(nd, "14:30"), gap)
            r["rth"] = C.ret(s, C.utc(nd, "13:30"), C.utc(nd, "20:00"), gap)
            r["post30_to_open"] = C.ret(s, t0 + C.mins(30), C.utc(nd, "13:30"), gap)
            r["post30_to_close"] = C.ret(s, t0 + C.mins(30), C.utc(nd, "20:00"), gap)
            r["day_before"] = C.ret(s, C.utc(d, "13:30"), t0, gap)
            r["v_ah"] = C.sum_between(df, "quote_vol", t0, C.utc(nd, "00:00"))
            r["v_dead"] = C.sum_between(df, "quote_vol", C.utc(nd, "00:00"), C.utc(nd, "08:00"))
            r["v_rth_prev"] = C.sum_between(df, "quote_vol", C.utc(d, "13:30"), t0)
            # funding: settlement at 00:00 UTC after the print vs the symbol's normal |rate|
            f0 = cast(pd.DataFrame, f[(ft >= C.utc(nd, "00:00") - C.mins(5)) &
                                      (ft <= C.utc(nd, "00:00") + C.mins(5))])
            f8 = cast(pd.DataFrame, f[(ft >= C.utc(nd, "08:00") - C.mins(5)) &
                                      (ft <= C.utc(nd, "08:00") + C.mins(5))])
            fund00 = float(C.col(f0, "rate").iloc[0]) if len(f0) else float("nan")
            r["fund00"] = fund00
            r["fund08"] = float(C.col(f8, "rate").iloc[0]) if len(f8) else float("nan")
            r["fund_norm_abs"] = norm_abs
            r["fund00_pctl"] = (float((rate.abs() <= abs(fund00)).mean())
                                if len(f) and np.isfinite(fund00) else float("nan"))
            r["fund_nonzero_share"] = float((rate != 0).mean()) if len(f) else float("nan")
            rows.append(r)
    out = pd.DataFrame(rows)
    for c in out.columns:
        if c not in ("sym", "date", "acc"):
            out[c] = pd.to_numeric(out[c])
    return out


def _cond(d: pd.DataFrame, sig: str, tgt: str, thr: float) -> str:
    g = cast(pd.DataFrame, d[C.col(d, sig).abs() > thr])
    if len(g) < 3:
        return f"n={len(g)}"
    x = cast(pd.Series, np.sign(C.col(g, sig)) * C.col(g, tgt))
    return f"n={len(g):3d} same-dir mean {x.mean() * 1e4:+.0f} bps med {x.median() * 1e4:+.0f} hit {(x > 0).mean():.2f}"


def amc_report(d: pd.DataFrame) -> None:
    print(f"\n== 3a. AMC earnings events: {len(d)} events, {d['sym'].nunique()} symbols ==")
    print("  " + ", ".join(f"{r.sym}:{r.date[5:]}@{r.acc}" for r in C.rows(d)))
    print("\n  size of moves (median |bps|):")
    for k in ["r5", "r15", "r30", "r60", "ah", "dead", "pre", "open1h", "rth"]:
        print(f"    {k:7s} med|r| {d[k].abs().median() * 1e4:6.0f}  mean|r| {d[k].abs().mean() * 1e4:6.0f}"
              f"  n {d[k].notna().sum()}")
    print(f"  AH volume / prior-RTH volume: median {(d['v_ah'] / d['v_rth_prev']).median():.2f}; "
          f"DEAD/prior-RTH {(d['v_dead'] / d['v_rth_prev']).median():.2f}")
    print("\n  how much of the AH move is done in the first 30 min?  |r30|/|ah| median "
          f"{(d['r30'].abs() / d['ah'].abs()).median():.2f}; sign agreement r30 vs ah "
          f"{(np.sign(d['r30']) == np.sign(d['ah'])).mean():.2f}")
    print("\n  regressions (slope, t):")
    for x, y in [("r30", "ah_post30"), ("r30", "dead"), ("r30", "post30_to_open"), ("r30", "post30_to_close"),
                 ("ah", "dead"), ("ah", "pre"), ("ah", "open1h"), ("ah", "rth"),
                 ("dead", "pre"), ("dead", "open1h"), ("dead", "rth"), ("pre", "open1h"), ("open1h", "rth")]:
        o = C.ols(C.col(d, x), C.col(d, y))
        print(f"    {y:15s} ~ {x:6s}: n={o.n:3d} slope {o.slope:+.3f} t={o.t:+.1f} R2={o.r2:.2f}")
    print("\n  conditional drift (enter in the direction of the first move):")
    for sig, tgt, thr in [("r30", "post30_to_open", 0.02), ("r30", "post30_to_open", 0.05),
                          ("r30", "post30_to_close", 0.02), ("r30", "post30_to_close", 0.05),
                          ("r30", "ah_post30", 0.02), ("r30", "dead", 0.02),
                          ("ah", "dead", 0.02), ("ah", "dead", 0.05), ("ah", "pre", 0.02),
                          ("ah", "open1h", 0.02), ("ah", "rth", 0.02),
                          ("dead", "pre", 0.01), ("dead", "open1h", 0.01), ("dead", "rth", 0.01)]:
        print(f"    sign({sig})*{tgt:15s} |{sig}|>{thr * 100:.0f}%: {_cond(d, sig, tgt, thr)}")
    print("\n  funding at the 00:00 / 08:00 UTC settlements after the print (rate in bps/8h):")
    print(f"    median |fund00| {d['fund00'].abs().median() * 1e4:.2f} vs symbol-normal median |rate| "
          f"{d['fund_norm_abs'].median() * 1e4:.2f}; |fund08| {d['fund08'].abs().median() * 1e4:.2f}; "
          f"max |fund00| {d['fund00'].abs().max() * 1e4:.1f}")
    print(f"    share of events with |fund00|>0: {(d['fund00'].abs() > 0).mean():.2f} vs symbol-normal "
          f"non-zero share {d['fund_nonzero_share'].median():.2f}; median within-symbol percentile of "
          f"|fund00| {d['fund00_pctl'].median():.2f}; events at >=90th pctl: {(d['fund00_pctl'] >= 0.9).sum()}/{len(d)}")
    o = C.ols(C.col(d, "ah"), C.col(d, "fund00"))
    sign_ah = cast(pd.Series, np.sign(C.col(d, "ah")))
    print(f"    fund00 ~ ah: slope {o.slope * 1e4:+.2f} bps per 100% (t {o.t:+.1f}); "
          f"corr(sign ah, fund00) {sign_ah.corr(C.col(d, 'fund00')):.2f}")
    print("\n  per event (bps): r30 | ah | dead | pre | open1h | rth | fund00")
    for r in C.rows(d.sort_values("date")):
        print(f"    {r.date} {r.sym:6s} {r.r30 * 1e4:+6.0f} {r.ah * 1e4:+6.0f} {r.dead * 1e4:+6.0f} "
              f"{r.pre * 1e4:+6.0f} {r.open1h * 1e4:+6.0f} {r.rth * 1e4:+6.0f} {r.fund00 * 1e4:+6.2f}")


def bmo_report(ev: pd.DataFrame) -> None:
    rows: List[Dict[str, float | str]] = []
    for sym_, g in ev[ev["kind"] == "BMO"].groupby("sym"):
        sym = cast(str, sym_)
        s = C.load_close(sym)
        for _, e in g.iterrows():
            t0 = C.as_ts(e["t0"])
            if t0 - C.days(3) < C.idx_min(s) or t0 + C.hours(8) > C.idx_max(s):
                continue
            d = str(t0.date())
            gap = C.mins(30)
            r: Dict[str, float | str] = {"sym": sym, "date": d, "acc": str(e["acc"])[11:16]}
            r["r15"] = C.ret(s, t0, t0 + C.mins(15), gap)
            r["r60"] = C.ret(s, t0, t0 + C.mins(60), gap)
            r["to_open"] = C.ret(s, t0, C.utc(d, "13:30"), gap)
            r["post15_to_open"] = C.ret(s, t0 + C.mins(15), C.utc(d, "13:30"), gap)
            r["open1h"] = C.ret(s, C.utc(d, "13:30"), C.utc(d, "14:30"), gap)
            r["rth"] = C.ret(s, C.utc(d, "13:30"), C.utc(d, "20:00"), gap)
            rows.append(r)
    d = pd.DataFrame(rows)
    for c in d.columns:
        if c not in ("sym", "date", "acc"):
            d[c] = pd.to_numeric(d[c])
    print(f"\n== 3b. BMO earnings events: {len(d)} (perp trades, cash pre-market only) ==")
    if len(d) < 3:
        return
    for k in ["r15", "r60", "to_open", "open1h", "rth"]:
        print(f"    {k:8s} med|r| {d[k].abs().median() * 1e4:6.0f}")
    for x, y in [("r15", "post15_to_open"), ("to_open", "open1h"), ("to_open", "rth")]:
        o = C.ols(C.col(d, x), C.col(d, y))
        print(f"    {y:15s} ~ {x:8s}: n={o.n} slope {o.slope:+.3f} t={o.t:+.1f}")
    for t in C.rows(d.sort_values("date")):
        print(f"    {t.date} {t.sym:6s} acc {t.acc} r15 {t.r15 * 1e4:+6.0f} to_open {t.to_open * 1e4:+6.0f} "
              f"open1h {t.open1h * 1e4:+6.0f} rth {t.rth * 1e4:+6.0f}")


def main() -> None:
    syms = C.universe(min_days=15)
    ev = classify(fetch_edgar(syms))
    print(f"EDGAR 8-K item 2.02 events in window: {len(ev)}  (AMC {(ev['kind'] == 'AMC').sum()}, "
          f"BMO {(ev['kind'] == 'BMO').sum()}, MID {(ev['kind'] == 'MID').sum()})")
    fund = pd.read_parquet(C.BN / "_funding.parquet")
    d = amc_rows(ev, fund)
    amc_report(d)
    liquid = {s: float(C.load_px(s)["quote_vol"].mean() * 288) for s in d["sym"].unique()}
    dl = cast(pd.DataFrame, d[C.col(d, "sym").map(liquid) > 5e6])
    print(f"\n== 3c. AMC subset ADV>$5M: {len(dl)} events ==")
    for sig, tgt, thr in [("r30", "post30_to_open", 0.02), ("r30", "post30_to_close", 0.02),
                          ("ah", "dead", 0.02), ("dead", "open1h", 0.01), ("dead", "rth", 0.01)]:
        print(f"    sign({sig})*{tgt:15s} |{sig}|>{thr * 100:.0f}%: {_cond(dl, sig, tgt, thr)}")
    bmo_report(ev)


if __name__ == "__main__":
    main()
