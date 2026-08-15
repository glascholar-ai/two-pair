#!/usr/bin/env python3
"""Binance TradFi perp vs Hyperliquid xyz stock perp: 5m spread, funding gap, lead-lag.

Inputs: data/bn5m/<SYM>USDT.parquet, data/bn5m/_funding.parquet,
        data/hl/xyz_<NAME>_5m.parquet, data/hl/xyz_<NAME>_funding.parquet,
        data/hl/universe.json (chosen name map from scan_hl_fetch.py).
Output: markdown tables on stdout + docs/scan/hl_*.csv.

Segments (UTC) for US names:  AH 20:00-24:00 | DEAD 00:00-08:00 | PRE 08:00-13:30 |
RTH 13:30-20:00 | WKND (Sat/Sun).  Korean names (SKHX->SKHYNIX, SMSN->SAMSUNG) use
KRX 00:00-06:30 | KR_AH 06:30-09:00 | KR_DEAD 09:00-23:00 | KR_PRE 23:00-24:00 | WKND.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
BN_DIR = ROOT / "data" / "bn5m"
HL_DIR = ROOT / "data" / "hl"
OUT_DIR = ROOT / "docs" / "scan"
KR_NAMES = ["SKHX", "SMSN"]
US_SEGS = ["RTH", "AH", "DEAD", "PRE", "WKND"]
KR_SEGS = ["KRX", "KR_AH", "KR_DEAD", "KR_PRE", "WKND"]
THRESHOLDS_BPS = (20.0, 50.0, 100.0)
# Fee assumptions (round trip, both legs): Binance taker 4 bps / maker 0; HL xyz taker
# fee is passed via HL_TAKER_BPS (growth-mode figure; see report for the doc source).
BN_TAKER_BPS = 4.0
HL_TAKER_BPS = 0.9


def col(df: pd.DataFrame, name: str) -> pd.Series:
    """Typed column accessor (pandas stubs return a union)."""
    return cast(pd.Series, df[name])


def load_chosen() -> List[Tuple[str, str]]:
    meta = json.loads((HL_DIR / "universe.json").read_text())
    return [(c["hl"], c["bn"]) for c in meta["chosen"]]


def segment_of(ts_utc: pd.Series, korean: bool) -> pd.Series:
    """Map bar open timestamps to a session-segment label."""
    dow = ts_utc.dt.dayofweek
    minute = ts_utc.dt.hour * 60 + ts_utc.dt.minute
    seg = pd.Series("", index=ts_utc.index, dtype=object)
    if korean:
        seg[minute < 390] = "KRX"
        seg[(minute >= 390) & (minute < 540)] = "KR_AH"
        seg[(minute >= 540) & (minute < 1380)] = "KR_DEAD"
        seg[minute >= 1380] = "KR_PRE"
    else:
        seg[minute < 480] = "DEAD"
        seg[(minute >= 480) & (minute < 810)] = "PRE"
        seg[(minute >= 810) & (minute < 1200)] = "RTH"
        seg[minute >= 1200] = "AH"
    seg[dow >= 5] = "WKND"
    return seg


def load_pair(hl: str, bn: str) -> Optional[pd.DataFrame]:
    """Aligned 5m frame: ts, bn_c, hl_c, bn_h/l, hl_h/l, trades, spread_bps, seg."""
    ph, pb = HL_DIR / f"xyz_{hl}_5m.parquet", BN_DIR / f"{bn}USDT.parquet"
    if not ph.exists() or not pb.exists():
        return None
    h = pd.read_parquet(ph).rename(columns={"c": "hl_c", "h": "hl_h", "l": "hl_l",
                                            "vol": "hl_vol", "trades": "hl_n"})
    b = pd.read_parquet(pb).rename(columns={"c": "bn_c", "h": "bn_h", "l": "bn_l",
                                            "quote_vol": "bn_qv", "trades": "bn_n"})
    df = h[["ts", "hl_c", "hl_h", "hl_l", "hl_vol", "hl_n"]].merge(
        b[["ts", "bn_c", "bn_h", "bn_l", "bn_qv", "bn_n"]], on="ts", how="inner")
    df = cast(pd.DataFrame, df[(df["hl_n"] > 0) & (df["bn_n"] > 0)]).copy()
    if len(df) < 500:
        return None
    df["dt"] = pd.to_datetime(col(df, "ts"), unit="ms", utc=True)
    df["spread_bps"] = np.log(col(df, "bn_c") / col(df, "hl_c")) * 1e4
    usdc_p = HL_DIR / "usdcusdt_5m.parquet"
    if usdc_p.exists():
        fx = pd.read_parquet(usdc_p)
        df = df.merge(fx, on="ts", how="left")
        df["usdcusdt"] = col(df, "usdcusdt").ffill()
        df["spread_adj_bps"] = col(df, "spread_bps") - np.log(col(df, "usdcusdt")) * 1e4
    else:
        df["spread_adj_bps"] = df["spread_bps"]
    df["seg"] = segment_of(col(df, "dt"), hl in KR_NAMES)
    df["r_bn"] = pd.Series(np.log(col(df, "bn_c"))).diff()
    df["r_hl"] = pd.Series(np.log(col(df, "hl_c"))).diff()
    gap = col(df, "ts").diff() != 300_000
    df.loc[gap, ["r_bn", "r_hl"]] = np.nan
    return cast(pd.DataFrame, df.reset_index(drop=True))


def half_life_bars(x: pd.Series) -> float:
    """AR(1) half-life in bars from lag-1 autocorrelation of a (demeaned) series."""
    x = x.dropna()
    if len(x) < 30:
        return float("nan")
    rho = float(x.autocorr(1))
    if not (0.0 < rho < 1.0):
        return 0.0 if rho <= 0 else float("inf")
    return float(-np.log(2.0) / np.log(rho))


def convergence_minutes(spread: pd.Series, thr: float) -> Tuple[int, float, float]:
    """Episodes where |spread| crosses above thr; minutes until it falls below thr/2.

    Returns (n_episodes, median_minutes, share_converged_within_60m).
    """
    a = spread.abs().to_numpy()
    n = len(a)
    durations: List[float] = []
    i = 1
    while i < n:
        if a[i] > thr and a[i - 1] <= thr:
            j = i
            while j < n and a[j] > thr / 2.0:
                j += 1
            durations.append(5.0 * (j - i) if j < n else float("inf"))
            i = j
        i += 1
    if not durations:
        return 0, float("nan"), float("nan")
    d = np.array(durations)
    return len(d), float(np.median(d)), float(np.mean(d <= 60.0))


def spread_stats(df: pd.DataFrame, hl: str, bn: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seg, g in df.groupby("seg"):
        s = col(g, "spread_bps")
        row: Dict[str, object] = {
            "hl": hl, "bn": bn, "seg": seg, "bars": len(g),
            "mean_bps": round(float(s.mean()), 2), "std_bps": round(float(s.std()), 2),
            "med_abs_bps": round(float(s.abs().median()), 2),
            "p99_abs_bps": round(float(s.abs().quantile(0.99)), 1),
            "max_abs_bps": round(float(s.abs().max()), 1),
            "adj_mean_bps": round(float(col(g, "spread_adj_bps").mean()), 2),
            "adj_std_bps": round(float(col(g, "spread_adj_bps").std()), 2),
            "rho1": round(float(s.autocorr(1)), 3) if len(s) > 30 else float("nan"),
            "hl_bars": round(half_life_bars(s), 1),
        }
        for thr in THRESHOLDS_BPS:
            row[f"p_gt{int(thr)}"] = round(float((s.abs() > thr).mean()), 4)
            n_ep, med, w60 = convergence_minutes(s, thr)
            row[f"conv{int(thr)}_n"] = n_ep
            row[f"conv{int(thr)}_med_min"] = med
            row[f"conv{int(thr)}_w60"] = round(w60, 2) if n_ep else float("nan")
        rows.append(row)
    return rows


def lead_lag(df: pd.DataFrame, hl: str) -> List[Dict[str, object]]:
    """Per segment: corr(r_bn[t-1], r_hl[t]) vs corr(r_hl[t-1], r_bn[t]) and OLS betas.

    OLS: r_hl[t] = a + b_bn * r_bn[t-1] + b_own * r_hl[t-1]  (and mirrored).
    A venue "leads" if its lagged return predicts the other's current return.
    """
    out: List[Dict[str, object]] = []
    d = df.assign(r_bn_l=df["r_bn"].shift(1), r_hl_l=df["r_hl"].shift(1))
    for seg, g in d.groupby("seg"):
        g = g.dropna(subset=["r_bn", "r_hl", "r_bn_l", "r_hl_l"])
        if len(g) < 100:
            continue
        b_hl_on_bn, t_hl_on_bn = _ols2(col(g, "r_hl"), col(g, "r_bn_l"), col(g, "r_hl_l"))
        b_bn_on_hl, t_bn_on_hl = _ols2(col(g, "r_bn"), col(g, "r_hl_l"), col(g, "r_bn_l"))
        out.append({
            "hl": hl, "seg": seg, "n": len(g),
            "corr_bn_lead": round(float(np.corrcoef(g["r_bn_l"], g["r_hl"])[0, 1]), 3),
            "corr_hl_lead": round(float(np.corrcoef(g["r_hl_l"], g["r_bn"])[0, 1]), 3),
            "corr_same": round(float(np.corrcoef(g["r_bn"], g["r_hl"])[0, 1]), 3),
            "beta_bn_lead": round(b_hl_on_bn, 3), "t_bn_lead": round(t_hl_on_bn, 1),
            "beta_hl_lead": round(b_bn_on_hl, 3), "t_bn_on_hl": round(t_bn_on_hl, 1),
        })
    return out


def _ols2(y: pd.Series, x1: pd.Series, x2: pd.Series) -> Tuple[float, float]:
    """OLS of y on [1, x1, x2]; return (beta_x1, t_x1) with HC0 standard errors."""
    # Work in bps to keep BLAS away from denormal territory.
    X = np.column_stack([np.ones(len(y)), x1.to_numpy(dtype=float) * 1e4,
                         x2.to_numpy(dtype=float) * 1e4])
    yv = y.to_numpy(dtype=float) * 1e4
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)
    meat = (X * (resid[:, None] ** 2)).T @ X
    cov = xtx_inv @ meat @ xtx_inv
    se = float(np.sqrt(cov[1, 1]))
    return float(beta[1]), float(beta[1] / se) if se > 0 else float("nan")


def funding_gap(hl: str, bn: str) -> Optional[Dict[str, object]]:
    """Annualized funding on each venue over the common window and the differential.

    Binance: 8h rate x 3 x 365.  HL xyz: hourly rate x 24 x 365 (summed to 8h buckets
    ending at each Binance funding time for the paired series).
    """
    fh_p = HL_DIR / f"xyz_{hl}_funding.parquet"
    if not fh_p.exists():
        return None
    fh = pd.read_parquet(fh_p)
    fb = pd.read_parquet(BN_DIR / "_funding.parquet")
    fb = fb[fb["symbol"] == f"{bn}USDT"].copy()
    fb["rate"] = fb["fundingRate"].astype(float)
    fb["ts"] = fb["fundingTime"].astype("int64")
    fb["bucket"] = (fb["ts"] // 28_800_000) * 28_800_000
    fh["bucket"] = ((fh["ts"] - 1) // 28_800_000) * 28_800_000 + 28_800_000
    hl8 = cast(pd.DataFrame, fh.groupby("bucket")["funding_rate"].agg(["sum", "count"]))
    hl8 = cast(pd.DataFrame, hl8.reset_index())
    hl8 = cast(pd.DataFrame, hl8[hl8["count"] == 8])
    m = cast(pd.DataFrame, fb[["bucket", "rate"]]).merge(
        cast(pd.DataFrame, hl8[["bucket", "sum"]]), on="bucket", how="inner")
    if len(m) < 20:
        return None
    m["diff"] = m["rate"] - m["sum"]           # bn minus hl, per 8h
    ann = 3 * 365 * 100.0                       # 8h rate -> % p.a.
    days = len(m) / 3.0
    last30 = m.tail(90)
    diff8 = m["diff"]
    sign = np.sign(diff8.mean())
    return {
        "hl": hl, "bn": bn, "days": round(days, 1),
        "bn_apr": round(float(m["rate"].mean() * ann), 2),
        "hl_apr": round(float(m["sum"].mean() * ann), 2),
        "diff_apr": round(float(diff8.mean() * ann), 2),
        "diff_apr_30d": round(float(last30["diff"].mean() * ann), 2),
        "diff_std_apr": round(float(diff8.std() * ann), 2),
        "diff_rho1": round(float(diff8.autocorr(1)), 3),
        "diff_rho3": round(float(diff8.autocorr(3)), 3),
        "same_sign_share": round(float((np.sign(diff8) == sign).mean()), 2),
        "bn_zero_share": round(float((m["rate"] == 0).mean()), 2),
        "bn_at_cap_share": round(float((m["rate"].abs() >= 0.02 - 1e-9).mean()), 3),
        "hl_at_base_share": round(float(np.isclose(fh["funding_rate"], 6.25e-6).mean()), 2),
    }


def carry_table(fund_rows: List[Dict[str, object]]) -> pd.DataFrame:
    """Net carry of long-cheap/short-rich after round-trip fees on both venues."""
    fee_rt_taker = 2 * BN_TAKER_BPS + 2 * HL_TAKER_BPS      # bps of one leg notional
    fee_rt_maker = 0.0 + 2 * HL_TAKER_BPS
    df = pd.DataFrame(fund_rows)
    df["gross_apr"] = df["diff_apr"].abs()
    df["direction"] = np.where(df["diff_apr"] > 0, "long HL / short BN", "long BN / short HL")
    df["fee_rt_taker_bps"] = fee_rt_taker
    df["breakeven_days_taker"] = (fee_rt_taker / 1e4) / (df["gross_apr"] / 100.0) * 365
    df["breakeven_days_maker"] = (fee_rt_maker / 1e4) / (df["gross_apr"] / 100.0) * 365
    df["net_apr_30d_hold_taker"] = df["gross_apr"] - fee_rt_taker / 1e4 * 100 * 365 / 30
    return df


def deviation_stats(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Spread minus its trailing-24h median (level-offset removed): pooled by segment.

    This isolates the tradeable, fast component of the cross-venue spread from the
    slow index-vs-oracle level offset.
    """
    rows: List[Dict[str, object]] = []
    for hl, df in frames.items():
        sp = col(df, "spread_bps")
        dev = sp - sp.rolling(288, min_periods=100).median().shift(1)
        d = df.assign(dev=dev).dropna(subset=["dev"])
        for seg, g in d.groupby("seg"):
            n20, med20, _ = convergence_minutes(col(g, "dev"), 20.0)
            rows.append({"hl": hl, "seg": seg, "bars": len(g),
                         "dev_std": float(g["dev"].std()),
                         "dev_p99": float(g["dev"].abs().quantile(0.99)),
                         "dev_hl_bars": half_life_bars(col(g, "dev")),
                         "p_gt10": float((g["dev"].abs() > 10).mean()),
                         "p_gt20": float((g["dev"].abs() > 20).mean()),
                         "p_gt50": float((g["dev"].abs() > 50).mean()),
                         "conv20_n": n20, "conv20_med_min": med20})
    return pd.DataFrame(rows)


def boundary_profile(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean |5m change in spread| and mean spread change by UTC hour, US names pooled.

    Detects predictable jumps at 08:00 / 13:30 / 20:00 / 00:00 boundaries.
    """
    parts: List[pd.DataFrame] = []
    for hl, df in frames.items():
        if hl in KR_NAMES:
            continue
        d = cast(pd.DataFrame, df[col(df, "dt").dt.dayofweek < 5]).copy()
        d["dspread"] = col(d, "spread_bps").diff()
        d.loc[col(d, "ts").diff() != 300_000, "dspread"] = np.nan
        dt = col(d, "dt")
        d["slot"] = (dt.dt.hour * 60 + dt.dt.minute) // 30 * 30
        parts.append(cast(pd.DataFrame, d[["slot", "dspread", "r_bn", "r_hl"]]))
    allp = pd.concat(parts).dropna()
    prof = allp.groupby("slot").agg(
        n=("dspread", "size"),
        abs_dspread=("dspread", lambda s: float(s.abs().mean())),
        mean_dspread=("dspread", "mean"),
        abs_r_bn_bps=("r_bn", lambda s: float(s.abs().mean() * 1e4)),
        abs_r_hl_bps=("r_hl", lambda s: float(s.abs().mean() * 1e4))).reset_index()
    prof["hhmm"] = prof["slot"].apply(lambda m: f"{int(m)//60:02d}:{int(m)%60:02d}")
    return prof.round(3)


def reference_stats(hl: str, bn: str, df: pd.DataFrame) -> List[Dict[str, object]]:
    """Reference-price layer by segment (hourly points where HL premium is known).

    bn_prem   = ln(bn_mark / bn_index)               Binance mark premium (bps)
    hl_prem   = HL hourly premium field (mid vs oracle, bps)
    idx_gap   = ln(bn_index / hl_oracle_est), hl_oracle_est = hl_close/(1+hl_prem)
    trade_gap = ln(bn_close / hl_close)   (same as spread_bps, for reference)
    """
    ref_p = HL_DIR / f"bnref_{bn}.parquet"
    fh_p = HL_DIR / f"xyz_{hl}_funding.parquet"
    if not ref_p.exists() or not fh_p.exists():
        return []
    ref = pd.read_parquet(ref_p)
    fh = pd.read_parquet(fh_p)
    fh["ts"] = (fh["ts"] // 3_600_000) * 3_600_000      # funding stamped ~hh:00:00.0xx
    d = df.merge(ref, on="ts", how="inner").merge(
        cast(pd.DataFrame, fh[["ts", "premium"]]), on="ts", how="inner")
    if d.empty:
        return []
    d["bn_prem"] = np.log(d["mark"] / d["index"]) * 1e4
    d["hl_prem"] = d["premium"] * 1e4
    d["idx_gap"] = np.log(d["index"] / (d["hl_c"] / (1.0 + d["premium"]))) * 1e4
    rows: List[Dict[str, object]] = []
    for seg, g in d.groupby("seg"):
        rows.append({"hl": hl, "seg": seg, "n_hours": len(g),
                     "idx_gap_mean": round(float(g["idx_gap"].mean()), 1),
                     "idx_gap_std": round(float(g["idx_gap"].std()), 1),
                     "bn_prem_mean": round(float(g["bn_prem"].mean()), 1),
                     "bn_prem_absmax": round(float(g["bn_prem"].abs().max()), 1),
                     "hl_prem_mean": round(float(g["hl_prem"].mean()), 1),
                     "hl_prem_absmax": round(float(g["hl_prem"].abs().max()), 1),
                     "trade_gap_mean": round(float(g["spread_bps"].mean()), 1)})
    return rows


def md_table(df: pd.DataFrame, cols: List[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n"
                   for _, r in df[cols].iterrows())
    return head + body


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chosen = load_chosen()
    sp_rows: List[Dict[str, object]] = []
    ll_rows: List[Dict[str, object]] = []
    fd_rows: List[Dict[str, object]] = []
    frames: Dict[str, pd.DataFrame] = {}
    for hl, bn in chosen:
        df = load_pair(hl, bn)
        if df is None:
            print(f"skip {hl}")
            continue
        frames[hl] = df
        sp_rows += spread_stats(df, hl, bn)
        ll_rows += lead_lag(df, hl)
        fr = funding_gap(hl, bn)
        if fr:
            fd_rows.append(fr)
    sp = pd.DataFrame(sp_rows)
    ll = pd.DataFrame(ll_rows)
    fd = carry_table(fd_rows)
    sp.to_csv(OUT_DIR / "hl_spread_by_segment.csv", index=False)
    ll.to_csv(OUT_DIR / "hl_leadlag_by_segment.csv", index=False)
    fd.to_csv(OUT_DIR / "hl_funding_gap.csv", index=False)
    span = pd.concat([f["dt"] for f in frames.values()])
    print(f"window {span.min()} .. {span.max()}  names={len(frames)}\n")
    # pooled segment summary (US names only, equal-weight by bar)
    us = cast(pd.DataFrame, sp[~sp["hl"].isin(KR_NAMES)])
    agg = us.groupby("seg").apply(lambda g: pd.Series({
        "names": len(g), "bars": int(g["bars"].sum()),
        "mean_bps": round(float(np.average(g["mean_bps"], weights=g["bars"])), 2),
        "std_bps": round(float(np.average(g["std_bps"], weights=g["bars"])), 2),
        "adj_mean": round(float(np.average(g["adj_mean_bps"], weights=g["bars"])), 2),
        "adj_std": round(float(np.average(g["adj_std_bps"], weights=g["bars"])), 2),
        "med_abs": round(float(np.average(g["med_abs_bps"], weights=g["bars"])), 2),
        "p99_abs": round(float(np.average(g["p99_abs_bps"], weights=g["bars"])), 1),
        "hl_bars": round(float(g["hl_bars"].replace(np.inf, np.nan).median()), 1),
        "p_gt20": round(float(np.average(g["p_gt20"], weights=g["bars"])), 4),
        "p_gt50": round(float(np.average(g["p_gt50"], weights=g["bars"])), 4),
        "p_gt100": round(float(np.average(g["p_gt100"], weights=g["bars"])), 4),
        "conv20_med_min": float(np.nanmedian(g["conv20_med_min"])),
        "conv50_med_min": float(np.nanmedian(g["conv50_med_min"])),
    })).reindex(US_SEGS).reset_index()
    print("## pooled US names by segment\n" + md_table(agg, list(agg.columns)))
    print("## per-name spread\n" + md_table(sp, [
        "hl", "seg", "bars", "mean_bps", "std_bps", "adj_mean_bps", "adj_std_bps",
        "med_abs_bps", "p99_abs_bps", "hl_bars", "p_gt20", "p_gt50", "p_gt100", "conv20_n", "conv20_med_min",
        "conv50_n", "conv50_med_min"]))
    llagg = ll[~ll["hl"].isin(KR_NAMES)].groupby("seg").agg(
        n=("n", "sum"), corr_bn_lead=("corr_bn_lead", "mean"),
        corr_hl_lead=("corr_hl_lead", "mean"), corr_same=("corr_same", "mean"),
        beta_bn_lead=("beta_bn_lead", "mean"), beta_hl_lead=("beta_hl_lead", "mean"),
        n_bn_sig=("t_bn_lead", lambda s: int((s > 2).sum())),
        n_hl_sig=("t_bn_on_hl", lambda s: int((s > 2).sum())),
        names=("hl", "count")).round(3).reindex(US_SEGS).reset_index()
    print("## lead-lag pooled\n" + md_table(llagg, list(llagg.columns)))
    print("## lead-lag per name\n" + md_table(ll, list(ll.columns)))
    dev = deviation_stats(frames)
    dev.to_csv(OUT_DIR / "hl_spread_deviation.csv", index=False)
    devagg = dev[~dev["hl"].isin(KR_NAMES)].groupby("seg").agg(
        bars=("bars", "sum"), dev_std=("dev_std", "median"), dev_p99=("dev_p99", "median"),
        dev_hl_bars=("dev_hl_bars", "median"), p_gt10=("p_gt10", "mean"),
        p_gt20=("p_gt20", "mean"), p_gt50=("p_gt50", "mean"),
        conv20_med_min=("conv20_med_min", "median")).round(3).reindex(US_SEGS).reset_index()
    print("## deviation from trailing-24h median, US names pooled\n"
          + md_table(devagg, list(devagg.columns)))
    devkr = cast(pd.DataFrame, dev[dev["hl"].isin(KR_NAMES)]).round(3)
    print("## deviation, KR names\n" + md_table(devkr, list(devkr.columns)))
    prof = boundary_profile(frames)
    prof.to_csv(OUT_DIR / "hl_boundary_profile.csv", index=False)
    print("## 30-min boundary profile (US names, weekdays)\n"
          + md_table(prof, ["hhmm", "n", "abs_dspread", "mean_dspread",
                            "abs_r_bn_bps", "abs_r_hl_bps"]))
    ref_rows: List[Dict[str, object]] = []
    for hl, df in frames.items():
        ref_rows += reference_stats(hl, dict(chosen)[hl], df)
    ref = pd.DataFrame(ref_rows)
    ref.to_csv(OUT_DIR / "hl_reference_layer.csv", index=False)
    refagg = ref[~ref["hl"].isin(KR_NAMES)].groupby("seg").agg(
        n_hours=("n_hours", "sum"), idx_gap_mean=("idx_gap_mean", "mean"),
        idx_gap_std=("idx_gap_std", "median"), bn_prem_mean=("bn_prem_mean", "mean"),
        bn_prem_absmax=("bn_prem_absmax", "max"), hl_prem_mean=("hl_prem_mean", "mean"),
        hl_prem_absmax=("hl_prem_absmax", "max"),
        trade_gap_mean=("trade_gap_mean", "mean")).round(1).reindex(US_SEGS).reset_index()
    print("## reference layer, US names pooled (bps)\n" + md_table(refagg, list(refagg.columns)))
    print("## reference layer, KR names\n"
          + md_table(cast(pd.DataFrame, ref[ref["hl"].isin(KR_NAMES)]), list(ref.columns)))
    print("## funding gap / carry\n" + md_table(fd, [
        "hl", "days", "bn_apr", "hl_apr", "diff_apr", "diff_apr_30d", "diff_std_apr",
        "diff_rho1", "same_sign_share", "bn_zero_share", "bn_at_cap_share",
        "hl_at_base_share", "direction", "breakeven_days_taker", "breakeven_days_maker"]))


if __name__ == "__main__":
    main()
