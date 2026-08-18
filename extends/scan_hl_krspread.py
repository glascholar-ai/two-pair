#!/usr/bin/env python3
"""Binance vs Hyperliquid Korean-stock perp spread: mean-reversion trade backtest.

Follow-up to scan_hl_spread.py, which adjudicated US names as no-space and flagged
the Korean legs (SKHX/SKHYNIX, SMSN/SAMSUNG) as the only pairs with tradeable
cross-venue noise (3x US std, half-life 4-13 bars). This script does the trade-level
work: episode anatomy + entry/exit simulation with fees and holding-period funding.

Signal: dev = spread - trailing-24h rolling median of spread,
        spread = ln(BN_close / HL_close) in bps, 5m bars, both venues traded.
Trade:  |dev| > thr at bar t  ->  enter at close of t+1, fade the spread
        (dev>0: short BN / long HL; dev<0: long BN / short HL);
        exit when |dev| <= exit_thr (observed t, filled t+1 close) or after 24h.
PnL:    side * (spread_exit - spread_entry)  [bps of one-leg notional]
        + funding accrued on both legs  -  round-trip fees (taker/taker 9.8 bps,
        BN-maker/HL-taker 1.8 bps; current promo schedules).

Outputs markdown tables on stdout + docs/scan/hl_krspread_trades.csv.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from scan_hl_spread import col, load_pair

ROOT = Path(__file__).parent
HL_DIR = ROOT / "data" / "hl"
BN_DIR = ROOT / "data" / "bn5m"
OUT_DIR = ROOT / "docs" / "scan"
PAIRS: List[Tuple[str, str]] = [("SKHX", "SKHYNIX"), ("SMSN", "SAMSUNG"),
                                ("SKHY", "SKHY")]
ANCHOR_BARS = 288                 # 24h of 5m bars
THRESHOLDS = (15.0, 20.0, 30.0)  # entry |dev| bps
EXIT_THR = 5.0                    # exit when |dev| back inside this
TIMEOUT_BARS = 288                # 24h max hold
FEE_TT = 9.8                      # taker both venues, round trip, bps sum of legs
FEE_MT = 1.8                      # BN maker 0 + HL taker 0.9 x2
KR_HOLIDAYS = {"2026-08-17"}      # KRX closed weekday inside sample (Liberation Day obs.)
FILTER_MS = 2 * 3_600_000         # funding-aware filter: projected window after entry
FILTER_MIN_BPS = -3.0             # skip entry if projected funding pnl below this


def build_frame(hl: str, bn: str) -> Optional[pd.DataFrame]:
    """Aligned 5m frame with dev = spread - trailing 24h median, holiday-relabeled."""
    df = load_pair(hl, bn)
    if df is None:
        return None
    sp = col(df, "spread_bps")
    df["anchor"] = sp.rolling(ANCHOR_BARS, min_periods=100).median().shift(1)
    df["dev"] = sp - col(df, "anchor")
    dt = col(df, "dt")
    hol = dt.dt.strftime("%Y-%m-%d").isin(KR_HOLIDAYS)
    df.loc[hol & (df["seg"] != "WKND"), "seg"] = "KR_HOL"
    return df


def bn_funding_series(bn: str) -> pd.DataFrame:
    """Binance funding events for one symbol: ts, rate."""
    fb = pd.read_parquet(BN_DIR / "_funding.parquet")
    fb = cast(pd.DataFrame, fb[fb["symbol"] == f"{bn}USDT"]).copy()
    out = pd.DataFrame({"ts": fb["fundingTime"].astype("int64"),
                        "rate": fb["fundingRate"].astype(float)})
    return out.sort_values("ts").reset_index(drop=True)


def hl_funding_series(hl: str) -> pd.DataFrame:
    """HL hourly funding events: ts, rate."""
    fh = pd.read_parquet(HL_DIR / f"xyz_{hl}_funding.parquet")
    out = pd.DataFrame({"ts": fh["ts"].astype("int64"),
                        "rate": fh["funding_rate"].astype(float)})
    return out.sort_values("ts").reset_index(drop=True)


def funding_pnl_bps(side: int, t0: int, t1: int,
                    fb: pd.DataFrame, fh: pd.DataFrame) -> float:
    """Funding PnL in bps for spread position over (t0, t1].

    side=+1 means long BN / short HL. Long leg pays positive funding.
    pnl = -side * sum(bn rates) + side * sum(hl rates), in bps.
    """
    bnr = float(fb.loc[(col(fb, "ts") > t0) & (col(fb, "ts") <= t1), "rate"].sum())
    hlr = float(fh.loc[(col(fh, "ts") > t0) & (col(fh, "ts") <= t1), "rate"].sum())
    return (-side * bnr + side * hlr) * 1e4


def simulate(df: pd.DataFrame, thr: float, fb: pd.DataFrame,
             fh: pd.DataFrame, hl: str, fund_filter: bool = False) -> pd.DataFrame:
    """One-position-at-a-time fade backtest; returns per-trade frame.

    fund_filter=True: skip entries whose funding pnl projected over the next
    FILTER_MS (using realized event rates as a proxy for the venue-published
    accruing rates a live system observes) is below FILTER_MIN_BPS.
    """
    ts = col(df, "ts").to_numpy(dtype="int64")
    sp = col(df, "spread_bps").to_numpy(dtype=float)
    dev = col(df, "dev").to_numpy(dtype=float)
    seg = col(df, "seg").to_numpy(dtype=object)
    r_bn = col(df, "r_bn").to_numpy(dtype=float)
    r_hl = col(df, "r_hl").to_numpy(dtype=float)
    n = len(df)
    trades: List[Dict[str, object]] = []
    i = ANCHOR_BARS
    while i < n - 2:
        if not (np.isfinite(dev[i]) and abs(dev[i]) > thr):
            i += 1
            continue
        side = -1 if dev[i] > 0 else 1          # fade: dev>0 -> short BN/long HL
        sig_i, ent_i = i, i + 1
        if fund_filter:
            proj = funding_pnl_bps(side, int(ts[ent_i]),
                                   int(ts[ent_i]) + FILTER_MS, fb, fh)
            if proj < FILTER_MIN_BPS:
                i += 1
                continue
        # widening window: back to last bar with |dev| < thr/2
        w0 = sig_i
        while w0 > ANCHOR_BARS and np.isfinite(dev[w0 - 1]) and abs(dev[w0 - 1]) >= thr / 2:
            w0 -= 1
        j = ent_i
        exit_reason = "censored"
        while j < n - 1:
            j += 1
            if np.isfinite(dev[j]) and abs(dev[j]) <= EXIT_THR:
                exit_reason = "converged"
                break
            if j - ent_i >= TIMEOUT_BARS:
                exit_reason = "timeout"
                break
        ex_i = min(j + 1, n - 1)                 # fill next close
        path = dev[ent_i:ex_i + 1]
        path = path[np.isfinite(path)]
        mae = float(np.max(side * -1 * (path - dev[ent_i]))) if len(path) else 0.0
        gross = side * (sp[ex_i] - sp[ent_i])
        fnd = funding_pnl_bps(side, int(ts[ent_i]), int(ts[ex_i]), fb, fh)
        cum_bn_w = float(np.nansum(r_bn[w0 + 1:sig_i + 1])) * 1e4
        cum_hl_w = float(np.nansum(r_hl[w0 + 1:sig_i + 1])) * 1e4
        cum_bn_h = float(np.nansum(r_bn[ent_i + 1:ex_i + 1])) * 1e4
        cum_hl_h = float(np.nansum(r_hl[ent_i + 1:ex_i + 1])) * 1e4
        trades.append({
            "hl": hl, "thr": thr, "seg": seg[sig_i],
            "t_sig": pd.Timestamp(int(ts[sig_i]), unit="ms", tz="UTC"),
            "dev_sig": round(float(dev[sig_i]), 1),
            "dev_ent": round(float(dev[ent_i]), 1), "side": side,
            "hold_min": (ex_i - ent_i) * 5, "exit": exit_reason,
            "gross_bps": round(float(gross), 1), "funding_bps": round(fnd, 2),
            "net_tt": round(float(gross) + fnd - FEE_TT, 1),
            "net_mt": round(float(gross) + fnd - FEE_MT, 1),
            "mae_bps": round(mae, 1),
            "widen_bn_bps": round(cum_bn_w, 1), "widen_hl_bps": round(cum_hl_w, 1),
            "hold_bn_bps": round(cum_bn_h, 1), "hold_hl_bps": round(cum_hl_h, 1),
        })
        i = ex_i + 1
    return pd.DataFrame(trades)


def agg_trades(tr: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Aggregate per-trade stats."""
    def stats(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n": len(g),
            "win_tt": round(float((col(g, "net_tt") > 0).mean()), 2),
            "med_gross": round(float(col(g, "gross_bps").median()), 1),
            "mean_net_tt": round(float(col(g, "net_tt").mean()), 1),
            "sum_net_tt": round(float(col(g, "net_tt").sum()), 0),
            "mean_net_mt": round(float(col(g, "net_mt").mean()), 1),
            "sum_net_mt": round(float(col(g, "net_mt").sum()), 0),
            "mean_fund": round(float(col(g, "funding_bps").mean()), 1),
            "med_hold_min": float(col(g, "hold_min").median()),
            "p_timeout": round(float((col(g, "exit") == "timeout").mean()), 2),
            "mae_p95": round(float(col(g, "mae_bps").quantile(0.95)), 1),
            "worst_net_tt": round(float(col(g, "net_tt").min()), 1),
        })
    out = tr.groupby(by).apply(stats).reset_index()
    return out


def fetch_1m(hl: str, bn: str) -> Optional[pd.DataFrame]:
    """1m closes both venues for HL's available window (~3.5 days); cached."""
    hp = HL_DIR / f"xyz_{hl}_1m.parquet"
    bp = HL_DIR / f"bn1m_{bn}.parquet"
    if not hp.exists():
        rows: List[Dict[str, Any]] = []
        end = int(time.time() * 1000)
        for _ in range(4):
            body = json.dumps({"type": "candleSnapshot",
                               "req": {"coin": f"xyz:{hl}", "interval": "1m",
                                       "startTime": end - 86_400_000, "endTime": end}}).encode()
            req = urllib.request.Request("https://api.hyperliquid.xyz/info", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    out = json.load(resp)
            except Exception as ex:  # noqa: BLE001
                print(f"  hl 1m {hl}: {ex!r}")
                break
            if isinstance(out, list) and out:
                rows.extend(out)
            end -= 86_400_000
            time.sleep(0.3)
        if not rows:
            return None
        raw = pd.DataFrame(rows)
        h1 = pd.DataFrame({"ts": raw["t"].astype("int64"), "hl_c": raw["c"].astype(float),
                           "hl_n": raw["n"].astype("int64")})
        h1 = h1.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        h1.to_parquet(hp, index=False)
    h1 = pd.read_parquet(hp)
    if not bp.exists():
        rows2: List[List[Any]] = []
        start = int(h1["ts"].min())
        while True:
            url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={bn}USDT"
                   f"&interval=1m&startTime={start}&limit=1500")
            try:
                with urllib.request.urlopen(url, timeout=20) as resp:
                    data = json.load(resp)
            except Exception as ex:  # noqa: BLE001
                print(f"  bn 1m {bn}: {ex!r}")
                break
            if not data:
                break
            rows2.extend(data)
            if len(data) < 1500:
                break
            start = int(data[-1][0]) + 1
            time.sleep(0.3)
        if not rows2:
            return None
        b1 = pd.DataFrame(rows2).iloc[:, [0, 4, 8]]
        b1.columns = ["ts", "bn_c", "bn_n"]
        b1 = b1.astype({"ts": "int64", "bn_c": float, "bn_n": "int64"})
        b1.drop_duplicates("ts").sort_values("ts").to_parquet(bp, index=False)
    b1 = pd.read_parquet(bp)
    m = h1.merge(b1, on="ts", how="inner")
    m = cast(pd.DataFrame, m[(m["hl_n"] > 0) & (m["bn_n"] > 0)]).copy()
    if len(m) < 500:
        return None
    m["spread_bps"] = np.log(col(m, "bn_c") / col(m, "hl_c")) * 1e4
    m["anchor"] = col(m, "spread_bps").rolling(1440, min_periods=300).median().shift(1)
    m["dev"] = col(m, "spread_bps") - col(m, "anchor")
    m["dt"] = pd.to_datetime(col(m, "ts"), unit="ms", utc=True)
    return cast(pd.DataFrame, m.reset_index(drop=True))


def one_min_episodes(m: pd.DataFrame, thr: float) -> pd.DataFrame:
    """1m dev episodes above thr: peak, duration above thr/2, minutes to re-enter 5bps."""
    dev = col(m, "dev").to_numpy(dtype=float)
    ts = col(m, "ts").to_numpy(dtype="int64")
    n = len(m)
    rows: List[Dict[str, object]] = []
    i = 1
    while i < n:
        if np.isfinite(dev[i]) and abs(dev[i]) > thr and (
                not np.isfinite(dev[i - 1]) or abs(dev[i - 1]) <= thr):
            j = i
            peak = abs(dev[i])
            while j < n - 1 and np.isfinite(dev[j]) and abs(dev[j]) > thr / 2:
                peak = max(peak, abs(dev[j]))
                j += 1
            k = j
            while k < n - 1 and np.isfinite(dev[k]) and abs(dev[k]) > EXIT_THR:
                k += 1
            rows.append({"t0": pd.Timestamp(int(ts[i]), unit="ms", tz="UTC"),
                         "peak": round(peak, 1),
                         "min_above_half": round((ts[j] - ts[i]) / 60_000, 0),
                         "min_to_5bps": round((ts[k] - ts[i]) / 60_000, 0)})
            i = k
        i += 1
    return pd.DataFrame(rows)


def volume_by_seg(df: pd.DataFrame, hl: str) -> pd.DataFrame:
    """Median 5m notional volume per venue by segment (capacity context)."""
    d = df.assign(hl_ntl=col(df, "hl_vol") * col(df, "hl_c"))
    out = d.groupby("seg").agg(
        bars=("ts", "size"),
        bn_qv_med=("bn_qv", "median"),
        hl_ntl_med=("hl_ntl", "median"),
        bn_n_med=("bn_n", "median"),
        hl_n_med=("hl_n", "median")).round(0).reset_index()
    out.insert(0, "hl", hl)
    return out


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n"
                   for _, r in df.iterrows())
    return head + body


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_trades: List[pd.DataFrame] = []
    filt_trades: List[pd.DataFrame] = []
    vols: List[pd.DataFrame] = []
    for hl, bn in PAIRS:
        df = build_frame(hl, bn)
        if df is None:
            print(f"skip {hl}")
            continue
        dtc = col(df, "dt")
        print(f"\n# {hl}/{bn}: {dtc.min()} .. {dtc.max()}  bars={len(df)}")
        fb, fh = bn_funding_series(bn), hl_funding_series(hl)
        vols.append(volume_by_seg(df, hl))
        for thr in THRESHOLDS:
            tr = simulate(df, thr, fb, fh, hl)
            if tr.empty:
                continue
            all_trades.append(tr)
            trf = simulate(df, thr, fb, fh, hl, fund_filter=True)
            if not trf.empty:
                filt_trades.append(trf)
    trades = pd.concat(all_trades).reset_index(drop=True)
    trades.to_csv(OUT_DIR / "hl_krspread_trades.csv", index=False)
    print("\n## by name x thr\n" + md(agg_trades(trades, ["hl", "thr"])))
    if filt_trades:
        tf = pd.concat(filt_trades).reset_index(drop=True)
        tf.to_csv(OUT_DIR / "hl_krspread_trades_filtered.csv", index=False)
        print("## funding-filtered (skip if projected 2h funding < -3 bps)\n"
              + md(agg_trades(tf, ["hl", "thr"])))
        tf20 = cast(pd.DataFrame, tf[tf["thr"] == 20.0])
        print("## filtered thr=20 by name x entry segment\n"
              + md(agg_trades(tf20, ["hl", "seg"])))
    t20 = cast(pd.DataFrame, trades[trades["thr"] == 20.0])
    print("## thr=20 by name x entry segment\n" + md(agg_trades(t20, ["hl", "seg"])))
    print("## thr=20 trades (worst 12 by net_tt)\n" + md(
        cast(pd.DataFrame, t20.sort_values("net_tt").head(12)[[
            "hl", "seg", "t_sig", "dev_sig", "side", "hold_min", "exit", "gross_bps",
            "funding_bps", "net_tt", "mae_bps"]])))
    print("## anatomy thr=20: |leg move| during widening / during hold (median bps)\n" + md(
        t20.groupby(["hl", "seg"]).apply(lambda g: pd.Series({
            "n": len(g),
            "widen_bn": round(float(col(g, "widen_bn_bps").abs().median()), 1),
            "widen_hl": round(float(col(g, "widen_hl_bps").abs().median()), 1),
            "hold_bn": round(float(col(g, "hold_bn_bps").abs().median()), 1),
            "hold_hl": round(float(col(g, "hold_hl_bps").abs().median()), 1),
        })).reset_index()))
    print("## 5m notional volume by segment (median, USD)\n" + md(pd.concat(vols)))
    for hl, bn in PAIRS[:1]:
        m = fetch_1m(hl, bn)
        if m is None:
            continue
        dtm = col(m, "dt")
        print(f"\n## 1m check {hl}: {dtm.min()} .. {dtm.max()}  bars={len(m)}")
        ep = one_min_episodes(m, 20.0)
        if len(ep):
            print(md(ep))


if __name__ == "__main__":
    main()
