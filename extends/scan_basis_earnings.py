#!/usr/bin/env python3
"""Q3: earnings-like after-hours events — perp vs IBKR AH tape at 1-min resolution.

Event = US trading day whose Binance perp AH (20:00-24:00 UTC) log move exceeds 4%.
For each event: Binance 1m klines (public API) and IBKR 1-min TRADES (SMART for
20:00-24:00, OVERNIGHT for 00:00-08:00), cached under data/ib/events/.
Outputs: lead-lag cross-correlation, time-to-X% of the AH move, DEAD-segment drift.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from scan_basis_prep import IBD, load_bn, merge_symbol, minute_of_day

EV_DIR = IBD / "events"
EV_DIR.mkdir(parents=True, exist_ok=True)
SYMS = ["MU", "NVDA", "TSLA", "AMD", "HOOD", "COIN", "PLTR", "MSTR", "SNDK", "WDC",
        "META", "AAPL", "ORCL", "GME", "SOFI", "RDDT", "MSFT", "JPM", "COST", "KO"]
THR = 0.04


def _ts(x: object) -> pd.Timestamp:
    """Narrow Timestamp arithmetic results (pandas stubs widen to Timestamp | NaT)."""
    return cast(pd.Timestamp, x)


def find_events(syms: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for s in syms:
        px = load_bn(s)["c"]
        idx = pd.DatetimeIndex(px.index)
        mod = minute_of_day(idx)
        lp = pd.Series(np.log(np.asarray(px.values, dtype=float)),
                       index=idx - pd.to_timedelta(mod, unit="m"))
        lp = cast(pd.Series, lp[mod >= 20 * 60])
        g = lp.groupby(level=0)
        r = g.last() - g.first()
        for d, v in r.items():
            day = pd.Timestamp(cast(Any, d))
            if abs(float(v)) > THR and day.dayofweek < 5:
                rows.append({"sym": s, "day": day, "ah_ret": float(v)})
    return pd.DataFrame(rows).sort_values(by=["day", "sym"]).reset_index(drop=True)


def bn_1m(sym: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    f = EV_DIR / f"{sym}_{start:%Y%m%d}_bn1m.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        return pd.Series(d["c"].values, index=pd.to_datetime(d["ts"], unit="ms", utc=True))
    rows: List[list] = []
    st = int(start.timestamp() * 1000)
    en = int(end.timestamp() * 1000)
    while st < en:
        url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}USDT&interval=1m"
               f"&startTime={st}&endTime={en}&limit=1500")
        with urllib.request.urlopen(url, timeout=20) as r:
            d = json.load(r)
        if not d:
            break
        rows.extend(d)
        st = int(d[-1][0]) + 60_000
        time.sleep(0.2)
    df = pd.DataFrame({"ts": [int(x[0]) for x in rows], "c": [float(x[4]) for x in rows]})
    df.to_parquet(f, index=False)
    return pd.Series(df["c"].values, index=pd.to_datetime(df["ts"], unit="ms", utc=True))


def ib_1m(ib: Any, sym: str, day: pd.Timestamp) -> pd.Series:
    """1-min last-trade closes 20:00 (day) -> 08:00 (day+1) UTC via SMART + OVERNIGHT."""
    from ib_insync import Stock  # type: ignore[import-untyped]  # ib_insync ships no stubs
    f = EV_DIR / f"{sym}_{day:%Y%m%d}_ib1m.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        return pd.Series(d["c"].values, index=pd.to_datetime(d["ts"], utc=True))
    out: List[Tuple[pd.Timestamp, float]] = []
    for exch, end, dur in (("SMART", _ts(day + pd.Timedelta(days=1)), "14400 S"),
                           ("OVERNIGHT", _ts(day + pd.Timedelta(days=1, hours=8)), "28800 S")):
        con = Stock(sym, exch, "USD")
        if not ib.qualifyContracts(con):
            continue
        end_s = end.strftime("%Y%m%d-%H:%M:%S")
        bars = ib.reqHistoricalData(con, endDateTime=end_s, durationStr=dur,
                                    barSizeSetting="1 min", whatToShow="TRADES",
                                    useRTH=False, formatDate=2, timeout=90)
        out.extend((_ts(pd.Timestamp(b.date)), float(b.close)) for b in bars)
        time.sleep(6)
    df = pd.DataFrame({"ts": [t for t, _ in out], "c": [c for _, c in out]})
    df = df.drop_duplicates("ts").sort_values(by="ts")
    df.to_parquet(f, index=False)
    return pd.Series(df["c"].values, index=pd.to_datetime(df["ts"], utc=True))


def xcorr(rp: pd.Series, rs: pd.Series, lags: range) -> Dict[int, float]:
    """corr(r_perp[t], r_stock[t+k]); k>0 => perp leads stock."""
    d = pd.DataFrame({"p": rp, "s": rs}).dropna()
    p = cast(pd.Series, d["p"])
    sser = cast(pd.Series, d["s"])
    return {k: float(p.corr(cast(pd.Series, sser.shift(-k)))) for k in lags}


def time_to_frac(path: pd.Series, total: float, frac: float) -> Optional[int]:
    """Minutes after 20:00 until |cum move| first reaches frac*|total| (same sign)."""
    if total == 0:
        return None
    cum = (path - path.iloc[0]) / total
    hit = cum[cum >= frac]
    if hit.empty:
        return None
    return int((hit.index[0] - path.index[0]).total_seconds() // 60)


def analyse_event(sym: str, day: pd.Timestamp, bn: pd.Series, ib: pd.Series) -> Dict[str, object]:
    t0 = day + pd.Timedelta(hours=20)
    t1 = day + pd.Timedelta(days=1)
    t2 = t1 + pd.Timedelta(hours=8)
    idx = pd.date_range(t0, t2, freq="1min", tz="UTC")
    lb = pd.Series(np.log(bn.reindex(idx).ffill().values.astype(float)), index=idx)
    ls = pd.Series(np.log(ib.reindex(idx).ffill().values.astype(float)), index=idx)
    ah = slice(t0, t1 - pd.Timedelta(minutes=1))
    rp, rs = cast(pd.Series, lb.diff()[ah]), cast(pd.Series, ls.diff()[ah])
    xc = xcorr(rp, rs, range(-5, 6))
    best = max(xc, key=lambda k: xc[k] if np.isfinite(xc[k]) else -9)
    lb_ah, ls_ah = cast(pd.Series, lb[ah]), cast(pd.Series, ls[ah])
    ah_perp = float(lb_ah.iloc[-1] - lb_ah.iloc[0])
    ah_stk = float(ls_ah.iloc[-1] - ls_ah.iloc[0])
    tot = ah_stk if abs(ah_stk) > 0 else ah_perp
    res: Dict[str, object] = {"sym": sym, "day": day.date(), "ah_perp%": ah_perp * 100, "ah_stock%": ah_stk * 100,
                              "xc_lag0": xc[0], "xc_best_lag": best, "xc_best": xc[best],
                              "xc_p_leads1": xc[1], "xc_s_leads1": xc[-1]}
    for fr in (0.5, 0.8):
        res[f"t{int(fr*100)}_perp"] = time_to_frac(lb_ah, tot, fr)
        res[f"t{int(fr*100)}_stock"] = time_to_frac(ls_ah, tot, fr)
    # basis at 23:59 and DEAD drift
    basis = lb - ls
    res["basis_2359_bps"] = float(cast(pd.Series, basis[ah]).iloc[-1] * 1e4)
    dead = slice(t1, t2 - pd.Timedelta(minutes=1))
    lb_dead = cast(pd.Series, lb[dead])
    ls_dead = cast(pd.Series, ls[dead]).dropna()
    b_dead = cast(pd.Series, basis[dead]).dropna()
    res["dead_perp%"] = float((lb_dead.iloc[-1] - lb_dead.iloc[0]) * 100)
    res["dead_stock%"] = float((ls_dead.iloc[-1] - ls_dead.iloc[0]) * 100) if len(ls_dead) > 10 else np.nan
    res["basis_0759_bps"] = float(b_dead.iloc[-1] * 1e4) if len(b_dead) else np.nan
    res["ib_ovn_bars"] = int(cast(pd.Series, ib[t1:t2]).shape[0])
    return res


def analyse_event_5m(sym: str, day: pd.Timestamp, m: pd.DataFrame) -> Dict[str, object]:
    """Fallback on cached 5m bars (SMART AH + OVERNIGHT): lead-lag at 5m lags, DEAD drift."""
    t0 = day + pd.Timedelta(hours=20)
    t1 = day + pd.Timedelta(days=1)
    t2 = t1 + pd.Timedelta(hours=8)
    idx = pd.date_range(t0, t2, freq="5min", tz="UTC")
    sub = m.reindex(idx)
    lb = pd.Series(np.log(arr_f(sub["bn"].ffill())), index=idx)
    ls = pd.Series(np.log(arr_f(sub["ib_last_ff"].ffill())), index=idx)
    ah = slice(t0, t1 - pd.Timedelta(minutes=5))
    rp, rs = cast(pd.Series, lb.diff()[ah]), cast(pd.Series, ls.diff()[ah])
    xc = xcorr(rp, rs, range(-3, 4))
    best = max(xc, key=lambda k: xc[k] if np.isfinite(xc[k]) else -9)
    lb_ah, ls_ah = cast(pd.Series, lb[ah]), cast(pd.Series, ls[ah])
    ah_perp = float(lb_ah.iloc[-1] - lb_ah.iloc[0])
    ah_stk = float(ls_ah.iloc[-1] - ls_ah.iloc[0])
    tot = ah_stk if abs(ah_stk) > 0 else ah_perp
    res: Dict[str, object] = {"sym": sym, "day": day.date(), "ah_perp%": ah_perp * 100,
                              "ah_stock%": ah_stk * 100, "xc_lag0": xc[0], "xc_best_lag_bars": best,
                              "xc_best": xc[best], "xc_p_leads1": xc[1], "xc_s_leads1": xc[-1]}
    for fr in (0.5, 0.8):
        res[f"t{int(fr*100)}_perp"] = time_to_frac(lb_ah, tot, fr)
        res[f"t{int(fr*100)}_stock"] = time_to_frac(ls_ah, tot, fr)
    basis = lb - ls
    res["basis_2355_bps"] = float(cast(pd.Series, basis[ah]).iloc[-1] * 1e4)
    dead = slice(t1, t2 - pd.Timedelta(minutes=5))
    lb_dead = cast(pd.Series, lb[dead])
    src_dead = cast(pd.Series, sub.loc[dead, "ib_src"])
    has_ovn = bool((src_dead == "OVERNIGHT").any())
    ls_dead = cast(pd.Series, ls[dead])
    res["dead_perp%"] = float((lb_dead.iloc[-1] - lb_dead.iloc[0]) * 100)
    res["dead_stock%"] = float((ls_dead.iloc[-1] - ls_dead.iloc[0]) * 100) if has_ovn else np.nan
    res["basis_0745_bps"] = float(cast(pd.Series, basis[dead]).iloc[-1] * 1e4) if has_ovn else np.nan
    # next-day RTH: does the AH move continue or revert? (13:30 -> 20:00 perp)
    nxt = m.reindex(pd.date_range(t1 + pd.Timedelta(hours=13, minutes=30),
                                  t1 + pd.Timedelta(hours=19, minutes=55), freq="5min", tz="UTC"))
    nb = arr_f(nxt["bn"].ffill())
    res["rth_next_perp%"] = float(np.log(nb[-1] / nb[0]) * 100) if np.isfinite(nb[0]) and np.isfinite(nb[-1]) else np.nan
    return res


def arr_f(x: object) -> np.ndarray:
    return np.asarray(cast(Any, x), dtype=float)


def main_5m(syms: List[str]) -> None:
    ev = find_events(syms)
    ev = cast(pd.DataFrame, ev[ev["day"] >= pd.Timestamp("2026-03-09", tz="UTC")])
    rows: List[Dict[str, object]] = []
    cache: Dict[str, pd.DataFrame] = {}
    for sym, day0 in zip(ev["sym"], ev["day"]):
        sym = str(sym)
        if sym not in cache:
            mm = merge_symbol(sym)
            if mm is None:
                continue
            cache[sym] = mm
        day = _ts(pd.Timestamp(cast(Any, day0)))
        rows.append(analyse_event_5m(sym, day, cache[sym]))
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent / "docs" / "scan" / "_earnings_events_5m.csv", index=False)
    print("## Q3 财报类 AH 事件 (perp |AH move| > 4%), 5m 分辨率 (IBKR SMART AH + OVERNIGHT)\n")
    print(str(df.round(2).to_markdown(index=False)) + "\n")
    num = df.select_dtypes("number")
    summ = pd.DataFrame({"mean": num.mean(), "median": num.median()}).T
    print("均值/中位:\n\n" + str(summ.round(2).to_markdown()) + "\n")
    lags = arr_f(df["xc_best_lag_bars"])
    print(f"最佳互相关滞后(5m bar): perp 领先 {(lags > 0).sum()} 例, 股票领先 {(lags < 0).sum()} 例, "
          f"同步 {(lags == 0).sum()} 例\n")


def main(argv: List[str]) -> None:
    if argv[1:2] == ["--5m"]:
        main_5m(argv[2:] or SYMS)
        return
    from ib_insync import IB  # type: ignore[import-untyped]
    ev = find_events(argv[1:] or SYMS)
    ev = ev[ev["day"] >= pd.Timestamp("2026-03-09", tz="UTC")]
    print(f"{len(ev)} events", file=sys.stderr)
    ib = IB()
    ib.connect("127.0.0.1", 7496, clientId=106, timeout=20, readonly=True)
    rows: List[Dict[str, object]] = []
    for sym, day0 in zip(ev["sym"], ev["day"]):
        sym = str(sym)
        day = _ts(pd.Timestamp(cast(Any, day0)))
        try:
            bn = bn_1m(sym, _ts(day + pd.Timedelta(hours=20)), _ts(day + pd.Timedelta(days=1, hours=8)))
            ibs = ib_1m(ib, sym, day)
        except Exception as ex:  # pylint: disable=broad-except
            print(f"{sym} {day.date()} failed: {ex!r}", file=sys.stderr)
            continue
        if ibs.empty:
            continue
        rows.append(analyse_event(sym, day, bn, ibs))
        print(f"done {sym} {day.date()}", file=sys.stderr)
    ib.disconnect()
    df = pd.DataFrame(rows)
    out_f = Path(__file__).parent / "docs" / "scan" / "_earnings_events.csv"
    df.to_csv(out_f, index=False)
    print("## Q3 财报类 AH 事件 (perp |AH move| > 4%)\n")
    print(str(df.round(2).to_markdown(index=False)) + "\n")
    num = df.select_dtypes("number")
    summ = pd.DataFrame({"mean": num.mean(), "median": num.median()}).T
    print("均值/中位:\n\n" + str(summ.round(2).to_markdown()) + "\n")
    lags = np.asarray(df["xc_best_lag"], dtype=float)
    print(f"最佳互相关滞后: perp 领先 {(lags > 0).sum()} 例, 股票领先 {(lags < 0).sum()} 例, "
          f"同步 {(lags == 0).sum()} 例\n")


if __name__ == "__main__":
    main(sys.argv)
