#!/usr/bin/env python3
"""Robustness + realistic-execution study of dead-zone (00-08 UTC) 5m spike reversal
in Binance US-equity perps.

Sections (all pooled over US-listed names, weekdays unless noted):
  A. close-to-close |z|>thr spikes: fwd 60/180m same-dir return by segment, hour,
     symbol, z-bin, direction, month; de-overlapped events.
  B. wick-based spikes (prev close -> bar extreme).
  C. resting-limit-order simulation: bid/ask at EWMA(30m) * exp(-/+ k*sigma),
     fill if bar wick crosses; exit at EWMA touch or 60/180m time stop (taker).
  D. tail: worst forward paths, AH-catalyst-day exclusion, oracle-wick events.
Run: python scan_offhours_reversal.py [--src bn|hl]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from session_anomaly_scan import NON_US, SEG

HERE = Path(__file__).parent
D_BN = HERE / "data" / "bn5m"
D_HL = HERE / "data" / "hl"
ASIA_LINKED = {"BABA", "EWY", "EWJ", "EWT", "TSM", "DRAM", "NVO", "ASML", "ARM", "SMH", "KORU",
               "EWZ"}
VOL_WIN = 288 * 3
FWD_H = (12, 36)  # 60m, 180m
TAKER_BPS = 4.0


def col(df: pd.DataFrame, c: str) -> pd.Series:
    return cast(pd.Series, df[c])


def dtidx(df: pd.DataFrame) -> pd.DatetimeIndex:
    return cast(pd.DatetimeIndex, df.index)


def tparts(idx: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hour, minute, dayofweek) arrays; goes through Series.dt for stub-friendliness."""
    s = idx.to_series()
    return (s.dt.hour.to_numpy(), s.dt.minute.to_numpy(), s.dt.dayofweek.to_numpy())


def load_ohlc(src: str, sym: str) -> Optional[pd.DataFrame]:
    p = D_BN / f"{sym}USDT.parquet" if src == "bn" else D_HL / f"xyz_{sym}_5m.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(col(df, "ts"), unit="ms", utc=True)
    df = cast(pd.DataFrame, df[~df.index.duplicated()]).sort_index()
    df = df[col(df, "c") > 0]
    return cast(pd.DataFrame, df[["o", "h", "l", "c"]]).astype(float)


def universe(src: str) -> List[str]:
    if src == "bn":
        syms = [p.stem[:-4] for p in D_BN.glob("*USDT.parquet")]
    else:
        syms = [p.stem[4:-3] for p in D_HL.glob("xyz_*_5m.parquet")]
    return sorted(s for s in syms if s not in NON_US and s not in ASIA_LINKED)


def segment_of(idx: pd.DatetimeIndex) -> pd.Series:
    hour, minute_, dow = tparts(idx)
    minute = hour * 60 + minute_
    seg = pd.Series("REST", index=idx)
    for name, a, b in SEG:
        seg[(minute >= a) & (minute < b)] = name
    seg[dow >= 5] = "WKND"
    return seg


def spike_events(df: pd.DataFrame, sym: str, thr: float = 3.0) -> pd.DataFrame:
    """Close-to-close and wick-based spikes with forward same-direction returns."""
    lc = cast(pd.Series, np.log(col(df, "c")))
    r = lc.diff()
    vol = r.rolling(VOL_WIN, min_periods=VOL_WIN // 2).std()
    z = r / vol
    up_w = cast(pd.Series, np.log(col(df, "h"))) - lc.shift(1)
    dn_w = cast(pd.Series, np.log(col(df, "l"))) - lc.shift(1)
    zw = pd.concat([up_w, -dn_w], axis=1).max(axis=1) / vol
    wick_dir = cast(pd.Series, np.sign(up_w + dn_w))  # which side went further
    out = pd.DataFrame({"sym": sym, "seg": segment_of(dtidx(df)), "z": z, "zw": zw,
                        "dir": np.sign(r), "wdir": wick_dir, "vol": vol,
                        "reject": (lc - lc.shift(1)) / (up_w.where(wick_dir > 0, dn_w))})
    for h in FWD_H:
        out[f"f{h}"] = lc.shift(-h) - lc
        # worst adverse excursion for a fader (same-dir max move within h bars)
    fmax = pd.concat([lc.shift(-k) - lc for k in range(1, 37)], axis=1)
    out["f36_max_same"] = fmax.max(axis=1)
    out["f36_min_same"] = fmax.min(axis=1)
    keep = (col(out, "z").abs() > thr) | (col(out, "zw") > thr)
    return cast(pd.DataFrame, out[keep]).dropna(subset=["z", "f36"])


def sd(ev: pd.DataFrame, c: str, dcol: str = "dir") -> pd.Series:
    return cast(pd.Series, ev[dcol] * ev[c] * 1e4)


def summarize(ev: pd.DataFrame, by: str, dcol: str = "dir") -> pd.DataFrame:
    e = ev.assign(s12=sd(ev, "f12", dcol), s36=sd(ev, "f36", dcol))
    g = e.groupby(by, observed=True)
    n = g.size()
    mean36, std36 = g["s36"].mean(), g["s36"].std()
    return pd.DataFrame({
        "n": n, "fwd60": g["s12"].mean(), "fwd180": mean36, "med180": g["s36"].median(),
        "revert%": g["s36"].apply(lambda v: (v < 0).mean() * 100),
        "t180": mean36 / std36 * np.sqrt(n),
    }).round(1)


def dedup(ev: pd.DataFrame, gap_bars: int = 36) -> pd.DataFrame:
    """Keep first event per symbol within a gap window (independent samples)."""
    keep = []
    for _, e in ev.groupby("sym"):
        last = None
        for t in e.index:
            if last is None or (t - last) > pd.Timedelta(minutes=5 * gap_bars):
                keep.append((e.loc[t, "sym"], t))
                last = t
    idx = pd.MultiIndex.from_tuples(keep)
    ev2 = ev.set_index("sym", append=True).swaplevel()
    return cast(pd.DataFrame, ev2.loc[idx]).reset_index(level=0)


def section_a(ev: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    e4 = cast(pd.DataFrame, ev[col(ev, "z").abs() > 4])
    out["by_seg"] = summarize(e4, "seg")
    dead = cast(pd.DataFrame, e4[e4["seg"] == "DEAD"]).copy()
    dead["hour"] = tparts(dtidx(dead))[0]
    out["by_hour"] = summarize(dead, "hour")
    dead["month"] = dtidx(dead).strftime("%m")
    out["by_month"] = summarize(dead, "month")
    dead["dirlab"] = np.where(dead["dir"] > 0, "up", "down")
    out["by_dir"] = summarize(dead, "dirlab")
    d3 = cast(pd.DataFrame, ev[(ev["seg"] == "DEAD")]).copy()
    d3["zbin"] = pd.cut(col(d3, "z").abs(), [3, 4, 5, 6, 8, 99],
                        labels=["3-4", "4-5", "5-6", "6-8", ">8"])
    out["by_zbin"] = summarize(d3.dropna(subset=["zbin"]), "zbin")
    ps = summarize(dead, "sym").sort_values("n", ascending=False)
    out["by_sym"] = ps
    dd = dedup(dead)
    dd["all"] = "DEAD dedup(3h)"
    out["dedup"] = summarize(dd, "all")
    return out


def section_b(ev: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Wick-based: spikes measured prev-close -> extreme; forward from close."""
    w = cast(pd.DataFrame, ev[(ev["zw"] > 4)]).copy()
    out: Dict[str, pd.DataFrame] = {"wick_by_seg": summarize(w, "seg", "wdir")}
    wd = cast(pd.DataFrame, w[w["seg"] == "DEAD"]).copy()
    wd["rejbin"] = pd.cut(col(wd, "reject").clip(-1, 1.5), [-1.01, 0.25, 0.5, 0.75, 1.51],
                          labels=["close<25%wick(full reject)", "25-50%", "50-75%", ">75%(closed near extreme)"])
    out["wick_by_reject"] = summarize(wd.dropna(subset=["rejbin"]), "rejbin", "wdir")
    wd["zwbin"] = pd.cut(col(wd, "zw"), [4, 5, 6, 8, 99], labels=["4-5", "5-6", "6-8", ">8"])
    out["wick_by_zw"] = summarize(wd.dropna(subset=["zwbin"]), "zwbin", "wdir")
    return out


def sim_limits(df: pd.DataFrame, sym: str, k: float, hold: int, halflife: int = 6,
               segs: Tuple[str, ...] = ("DEAD",)) -> pd.DataFrame:
    """Resting bid/ask at ewma*exp(-/+k*sigma); one position per side at a time.

    Fill: bar low < bid level (buy) / high > ask level (sell), level from prior bar.
    Exit: first later bar whose close is back through the *current* EWMA (taker at
    close) or close of bar entry+hold. Returns per-trade rows with gross log bps.
    """
    lc = np.log(col(df, "c").to_numpy(dtype=float))
    hi = np.log(col(df, "h").to_numpy(dtype=float))
    lo = np.log(col(df, "l").to_numpy(dtype=float))
    r = np.diff(lc, prepend=np.nan)
    sig = pd.Series(r).rolling(VOL_WIN, min_periods=VOL_WIN // 2).std().to_numpy(dtype=float)
    ew = pd.Series(lc).ewm(halflife=halflife).mean().to_numpy(dtype=float)
    seg = segment_of(dtidx(df)).to_numpy()
    ok = np.isin(seg, list(segs)) & ~np.isnan(sig)
    n = len(lc)
    ew_prev = np.roll(ew, 1)
    sig_prev = np.roll(sig, 1)
    rows: List[Dict[str, object]] = []
    for side in (1, -1):
        lvl_all = ew_prev - side * k * sig_prev
        cand = ok & ((lo < lvl_all) if side > 0 else (hi > lvl_all))
        cand[:1] = False
        cand[n - hold - 1:] = False
        cands = np.flatnonzero(cand)
        last_exit = -1
        for i in cands:
            if i <= last_exit:
                continue
            entry = lvl_all[i]
            j_exit, reason = i + hold, "time"
            for j in range(i + 1, i + hold + 1):
                if (side > 0 and lc[j] >= ew[j]) or (side < 0 and lc[j] <= ew[j]):
                    j_exit, reason = j, "ewma"
                    break
            pnl = side * (lc[j_exit] - entry) * 1e4
            if side > 0:
                mae = (lo[i + 1:j_exit + 1].min() - entry) * 1e4
            else:
                mae = (entry - hi[i + 1:j_exit + 1].max()) * 1e4
            rows.append({"sym": sym, "t": dtidx(df)[i], "side": side, "k": k, "hold": hold,
                         "seg": seg[i], "gross": pnl, "mae": min(mae, 0.0), "reason": reason,
                         "bars": j_exit - i, "sig_bps": sig_prev[i] * 1e4})
            last_exit = j_exit
    return pd.DataFrame(rows)


def trade_stats(tr: pd.DataFrame, notional: float = 20_000.0) -> pd.Series:
    net = col(tr, "gross") - TAKER_BPS  # maker in, taker out
    ts = col(tr, "t")
    days = max((ts.max() - ts.min()).days, 1) if len(tr) else 1
    daily = (net * notional / 1e4).groupby(ts.dt.floor("D")).sum()
    dd = (daily.cumsum() - daily.cumsum().cummax()).min() if len(daily) else 0.0
    return pd.Series({
        "fills": len(tr), "fills/day": len(tr) / days,
        "gross_bps": tr["gross"].mean(), "net_bps": net.mean(), "med_net": net.median(),
        "win%": (net > 0).mean() * 100, "p5_net": net.quantile(0.05), "min_net": net.min(),
        "ewma_exit%": (tr["reason"] == "ewma").mean() * 100,
        "$/day@20k": daily.mean() if len(daily) else 0.0,
        "daily_sharpe": daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 2 else 0.0,
        "maxDD_$": dd, "t_stat": net.mean() / net.std() * np.sqrt(len(net)) if len(net) > 2 else 0.0,
    })


def section_c(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grid = []
    all_trades: List[pd.DataFrame] = []
    for k in (2.0, 3.0, 4.0):
        for hold in (12, 36):
            trs = [sim_limits(df, s, k, hold, segs=("DEAD", "AH", "PRE", "OPEN", "REST", "WKND"))
                   for s, df in data.items()]
            tr = pd.concat([t for t in trs if len(t)])
            all_trades.append(tr)
            assert isinstance(tr, pd.DataFrame)
            for sg, e in tr.groupby("seg"):
                st = trade_stats(e)
                st["k"], st["hold"], st["seg"] = k, hold, sg
                grid.append(st)
    g = pd.DataFrame(grid).set_index(["k", "hold", "seg"])
    return g, pd.concat(all_trades)


def section_d(ev: pd.DataFrame, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    dead = cast(pd.DataFrame, ev[(ev["seg"] == "DEAD") & (col(ev, "z").abs() > 4)]).copy()
    s36 = sd(dead, "f36")
    q = s36.quantile([0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])
    out: Dict[str, pd.DataFrame] = {"quantiles": q.round(0).to_frame("same-dir fwd180 bps")}
    worst = dead.assign(sd36=s36).sort_values("sd36", ascending=False).head(15)
    out["worst15"] = cast(pd.DataFrame, worst[["sym", "z", "sd36"]]).assign(
        max_adverse=(worst["dir"] * worst["f36_max_same"] * 1e4)).round(1)
    # AH catalyst flag: preceding AH (20:00-24:00 same UTC evening) |move| > 3% or |z|>6
    flags = set()
    for s, df in data.items():
        lc = cast(pd.Series, np.log(col(df, "c")))
        r = lc.diff()
        z = r / r.rolling(VOL_WIN, min_periods=VOL_WIN // 2).std()
        seg = segment_of(dtidx(df))
        ah = seg == "AH"
        day = dtidx(df).to_series().dt.floor("D")
        ahret = cast(pd.Series, cast(pd.Series, lc[ah]).groupby(day[ah]).agg(
            lambda v: v.iloc[-1] - v.iloc[0]))
        ahz = cast(pd.Series, cast(pd.Series, z[ah]).abs().groupby(day[ah]).max())
        bad = cast(pd.Series, ahret[(ahret.abs() > 0.03) | (ahz > 6)])
        for d_ in pd.DatetimeIndex(bad.index):  # AH of day d_ precedes DEAD of d_+1
            flags.add((s, (d_ + pd.Timedelta(days=1)).date()))
    dead["ah_flag"] = [(s, t.date()) in flags for s, t in zip(dead["sym"], dtidx(dead))]
    dead["lab"] = np.where(dead["ah_flag"], "AH-catalyst day", "clean day")
    out["ah_split"] = summarize(dead, "lab")
    # oracle-wick style: wick > 3% beyond prev close but close rejects >75% of it
    ow = cast(pd.DataFrame, ev[(col(ev, "seg").isin(["DEAD", "WKND", "AH", "PRE"]))
                               & (ev["zw"] > 6) & (ev["reject"] < 0.25)]).copy()
    ow["wick_bps"] = ow["zw"] * ow["vol"] * 1e4
    ow = cast(pd.DataFrame, ow[ow["wick_bps"] > 300])
    out["oracle_wicks"] = cast(pd.DataFrame, ow.groupby("seg").agg(n=("sym", "size"),
                                                syms=("sym", lambda v: ",".join(sorted(set(v))[:12])),
                                                med_wick_bps=("wick_bps", "median"),
                                                max_wick_bps=("wick_bps", "max")))
    out["oracle_top"] = cast(pd.DataFrame, ow.sort_values("wick_bps", ascending=False).head(12)[
        ["sym", "seg", "wick_bps", "reject", "z"]]).round(2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="bn", choices=["bn", "hl"])
    ap.add_argument("--min-days", type=int, default=30)
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    data: Dict[str, pd.DataFrame] = {}
    for s in universe(a.src):
        df = load_ohlc(a.src, s)
        if df is not None and len(df) >= a.min_days * 288:
            data[s] = df
    print(f"src={a.src} symbols={len(data)} span "
          f"{min(d.index[0] for d in data.values()):%Y-%m-%d} .. "
          f"{max(d.index[-1] for d in data.values()):%Y-%m-%d}")
    ev = pd.concat([spike_events(df, s) for s, df in data.items()])
    print("\n########## A. close-to-close spikes ##########")
    for k, v in section_a(ev).items():
        print(f"\n-- {k} --")
        print(v.head(40).to_string() if k == "by_sym" else v.to_string())
        if k == "by_sym":
            print(f"symbols with negative fwd180: {(v['fwd180'] < 0).sum()}/{len(v)}; "
                  f"top-5 names' event share: {v['n'].head(5).sum() / v['n'].sum():.2f}")
    print("\n########## B. wick spikes ##########")
    for k, v in section_b(ev).items():
        print(f"\n-- {k} --")
        print(v.to_string())
    print("\n########## C. resting limit-order simulation ##########")
    grid, trades = section_c(data)
    print(grid.round(1).to_string())
    print("\n########## D. tail risk ##########")
    for k, v in section_d(ev, data).items():
        print(f"\n-- {k} --")
        print(v.to_string())
    out = HERE / "data" / "hl" / f"offhours_trades_{a.src}.parquet"
    trades.to_parquet(out, index=False)
    print("\ntrades saved:", out)


if __name__ == "__main__":
    main()
