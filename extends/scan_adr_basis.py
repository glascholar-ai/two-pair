#!/usr/bin/env python3
"""Q1/Q2: US-listed perp vs home-line during the home session.

Per pair: basis stats (mean/std/AR1 half-life), contemporaneous beta of perp
returns on home returns (5m & 30m), lead-lag cross-correlations, basis
mean-reversion trade simulation, and the "snap" regressions at 07:55->08:30 and
13:25->13:45 UTC.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd

from scan_adr_common import (PAIRS_ADR, Pair, build_frame, col, days, dow, dtidx, half_life, lg,
                             minute_of_day, ols)

BPS = 1e4


def _roll_mean(b: pd.Series, win: int) -> pd.Series:
    """Rolling mean over the last `win` *in-session* bars (NaN gaps skipped)."""
    bs = b.dropna()
    r = pd.Series(bs.rolling(win, min_periods=win // 4).mean(), index=bs.index)
    return r.reindex(b.index)


def _roll_std(b: pd.Series, win: int) -> pd.Series:
    bs = b.dropna()
    r = pd.Series(bs.rolling(win, min_periods=win // 4).std(), index=bs.index)
    return r.reindex(b.index)


def basis_stats(df: pd.DataFrame) -> Dict[str, float]:
    b = col(df.loc[col(df, "sess")], "basis").dropna()
    # remove per-day mean to isolate intraday dispersion
    day = b.groupby(days(dtidx(b)))
    b_intra = b - day.transform("mean")
    daily_mean = day.mean()
    hl = half_life(b)
    hl_intra = half_life(b_intra)
    rp = lg(col(df, "perp")).diff().where(col(df, "sess"))
    rh = lg(col(df, "home_usd")).diff().where(col(df, "sess"))
    return {
        "perp_ret_ar1": float(rp.autocorr(1)),
        "home_ret_ar1": float(rh.autocorr(1)),
        "n_bars": float(len(b)),
        "n_days": float(days(dtidx(b)).nunique()),
        "mean_bps": float(b.mean() * BPS),
        "std_bps": float(b.std() * BPS),
        "intra_std_bps": float(b_intra.std() * BPS),
        "daily_mean_std_bps": float(daily_mean.std() * BPS),
        "ar1_5m": float(b.autocorr(1)) if len(b) > 30 else float("nan"),
        "ar1_intra_5m": float(b_intra.autocorr(1)) if len(b) > 30 else float("nan"),
        "hl_bars": hl,
        "hl_intra_bars": hl_intra,
        "abs_gt_50bps_pct": float((b.abs() > 50e-4).mean() * 100),
        "abs_gt_100bps_pct": float((b.abs() > 100e-4).mean() * 100),
        "p99_abs_bps": float(b.abs().quantile(0.99) * BPS),
    }


def _sess_returns(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """k-bar log returns of perp / home, only where both endpoints in same session-day."""
    lp = lg(col(df, "perp"))
    lh = lg(col(df, "home_usd"))
    rp = lp - lp.shift(k)
    rh = lh - lh.shift(k)
    sess = col(df, "sess")
    ok = sess & sess.shift(k, fill_value=False)
    day = pd.Series(days(dtidx(df)), index=df.index)
    ok &= day == day.shift(k)
    return pd.DataFrame({"rp": rp[ok], "rh": rh[ok]}).dropna()


def betas(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, tag in [(1, "5m"), (6, "30m")]:
        r = _sess_returns(df, k)
        b, t, r2, n = ols(col(r, "rp"), col(r, "rh"))
        out[f"beta_{tag}"] = b
        out[f"t_{tag}"] = t
        out[f"r2_{tag}"] = r2
        out[f"n_{tag}"] = float(n)
        out[f"perp_std_{tag}_bps"] = float(r["rp"].std() * BPS)
        out[f"home_std_{tag}_bps"] = float(r["rh"].std() * BPS)
    return out


def lead_lag(df: pd.DataFrame, max_lag: int = 6) -> pd.Series:
    """corr(perp_ret_t, home_ret_{t-k}) for k in -max_lag..max_lag (k>0: home leads)."""
    lp = lg(col(df, "perp"))
    lh = lg(col(df, "home_usd"))
    rp = lp.diff()
    rh = lh.diff()
    sess = col(df, "sess")
    day = pd.Series(days(dtidx(df)), index=df.index)
    out: Dict[int, float] = {}
    for k in range(-max_lag, max_lag + 1):
        x = rh.shift(k)
        ok = sess & sess.shift(k, fill_value=False) & (day == day.shift(k))
        d = pd.DataFrame({"a": rp[ok], "b": x[ok]}).dropna()
        out[k] = float(d.corr().iloc[0, 1]) if len(d) > 30 else float("nan")
    return pd.Series(out)


def predictive(df: pd.DataFrame) -> Dict[str, float]:
    """Perp next-return regressed on (a) basis deviation, (b) last home 5m return."""
    lp = lg(col(df, "perp"))
    lh = lg(col(df, "home_usd"))
    sess = col(df, "sess")
    day = pd.Series(days(dtidx(df)), index=df.index)
    b = col(df, "basis").where(sess)
    dev = b - _roll_mean(b, 78)
    out: Dict[str, float] = {}
    for k, tag in [(1, "5m"), (6, "30m")]:
        fwd_p = lp.shift(-k) - lp
        fwd_h = lh.shift(-k) - lh
        ok = sess & sess.shift(-k, fill_value=False) & (day == day.shift(-k))
        bb, tt, _, n = ols(fwd_p[ok], dev[ok])
        out[f"perp_fwd{tag}_on_dev_b"] = bb
        out[f"perp_fwd{tag}_on_dev_t"] = tt
        bh, th, _, _ = ols(fwd_h[ok], dev[ok])
        out[f"home_fwd{tag}_on_dev_b"] = bh
        out[f"home_fwd{tag}_on_dev_t"] = th
        # last-5m home return -> perp forward return (lag trade)
        rh1 = lh.diff()
        rp1 = lp.diff()
        bl, tl, _, _ = ols(fwd_p[ok], cast(pd.Series, rh1[ok]))
        out[f"perp_fwd{tag}_on_home_lag5m_b"] = bl
        out[f"perp_fwd{tag}_on_home_lag5m_t"] = tl
        diff1 = rh1 - rp1
        bq, tq, _, _ = ols(fwd_p[ok], diff1[ok])
        out[f"perp_fwd{tag}_on_(home-perp)_5m_b"] = bq
        out[f"perp_fwd{tag}_on_(home-perp)_5m_t"] = tq
        out[f"n_{tag}"] = float(n)
    return out


def mr_trade(df: pd.DataFrame, k_in: float = 2.0, k_out: float = 0.5,
             win: int = 78) -> Dict[str, float]:
    """Basis mean-reversion: enter |dev|>k_in*std, exit |dev|<k_out*std or session end.

    PnL is quoted as bps of one leg: 'pair' = change in basis captured
    (long perp/short home or reverse); 'perp_only' = perp leg only.
    """
    sess = col(df, "sess")
    b = col(df, "basis").where(sess)
    mu = _roll_mean(b, win)
    sd = _roll_std(b, win)
    dev = (b - mu)
    z = dev / sd
    lp = lg(col(df, "perp"))
    day = pd.Series(days(dtidx(df)), index=df.index)
    trades: List[Dict[str, float]] = []
    pos = 0
    e_b = e_p = 0.0
    e_t = df.index[0]
    zz, bb, pp = z.to_numpy(dtype=float), b.to_numpy(dtype=float), lp.to_numpy(dtype=float)
    ss, dd, sdv = sess.to_numpy(dtype=bool), day.to_numpy(), sd.to_numpy(dtype=float)
    idx = dtidx(df)
    for i in range(len(df)):
        if pos != 0:
            end_sess = (not ss[i]) or dd[i] != dd[i - 1]
            exit_now = end_sess or (not np.isnan(zz[i]) and abs(zz[i]) < k_out)
            if exit_now:
                j = i if not end_sess else i - 1
                pair_pnl = -pos * (bb[j] - e_b) if not np.isnan(bb[j]) else 0.0
                perp_pnl = -pos * (pp[j] - e_p)
                trades.append({"pair_bps": pair_pnl * BPS, "perp_bps": perp_pnl * BPS,
                               "hold_min": (idx[j] - e_t).total_seconds() / 60,
                               "forced": float(end_sess)})
                pos = 0
        if pos == 0 and ss[i] and not np.isnan(zz[i]) and abs(zz[i]) > k_in and sdv[i] > 0:
            pos = int(np.sign(zz[i]))    # +1: perp rich -> short perp / long home
            e_b, e_p, e_t = bb[i], pp[i], idx[i]
    if not trades:
        return {"n_trades": 0.0}
    t = pd.DataFrame(trades)
    n_days = float(days(dtidx(sess[sess])).nunique())
    return {
        "n_trades": float(len(t)),
        "per_day": float(len(t) / max(n_days, 1)),
        "pair_mean_bps": float(t["pair_bps"].mean()),
        "pair_med_bps": float(t["pair_bps"].median()),
        "pair_win_pct": float((t["pair_bps"] > 0).mean() * 100),
        "pair_t": float(t["pair_bps"].mean() / t["pair_bps"].std() * np.sqrt(len(t))),
        "perp_mean_bps": float(t["perp_bps"].mean()),
        "perp_win_pct": float((t["perp_bps"] > 0).mean() * 100),
        "perp_t": float(t["perp_bps"].mean() / t["perp_bps"].std() * np.sqrt(len(t))),
        "hold_med_min": float(t["hold_min"].median()),
        "forced_pct": float(t["forced"].mean() * 100),
        "entry_sd_bps_med": float(np.nanmedian(sdv[ss]) * BPS),
    }


def _at(df: pd.DataFrame, name: str, hh: int, mm: int) -> pd.Series:
    m = minute_of_day(dtidx(df)) == hh * 60 + mm
    s = col(df.loc[m], name)
    return pd.Series(s.to_numpy(), index=days(dtidx(s)))


def _last_home_basis(df: pd.DataFrame) -> pd.Series:
    """Per day: last in-session basis (perp vs last home print), and its time."""
    b = col(df.loc[col(df, "sess")], "basis").dropna()
    return b.groupby(days(dtidx(b))).last()


def snap(df: pd.DataFrame, p: Pair) -> Dict[str, float]:
    """Regress perp returns across the 08:00 / 13:30 snaps on the pre-snap basis.

    basis0 = ln(perp(t0) / last home print at-or-before t0) — no look-ahead.
    """
    out: Dict[str, float] = {}
    sess = col(df, "sess")
    home_ff = col(df, "home_usd").where(sess).ffill(limit=288)   # last home print, <=24h stale
    for t0, t1, tag in [((7, 55), (8, 30), "0755_0830"), ((7, 55), (13, 45), "0755_1345"),
                        ((13, 25), (13, 45), "1325_1345"), ((13, 25), (14, 30), "1325_1430")]:
        p0 = _at(df, "perp", *t0)
        p1 = _at(df, "perp", *t1)
        h0 = _at(pd.DataFrame({"h": home_ff}), "h", *t0)
        ret = np.log(p1 / p0)
        basis0 = np.log(p0 / h0.reindex(p0.index))
        d0 = pd.DataFrame({"r": ret, "b": basis0}).dropna()
        d = cast(pd.DataFrame, d0[dow(dtidx(d0)) < 5])
        rr, bs = col(d, "r"), col(d, "b")
        b, t, r2, n = ols(rr, bs)
        out[f"{tag}_beta"] = b
        out[f"{tag}_t"] = t
        out[f"{tag}_r2"] = r2
        out[f"{tag}_n"] = float(n)
        out[f"{tag}_basis0_std_bps"] = float(d["b"].std() * BPS)
        out[f"{tag}_ret_std_bps"] = float(d["r"].std() * BPS)
        fade = -np.sign(bs - float(bs.median())) * rr
        out[f"{tag}_fade_mean_bps"] = float(fade.mean() * BPS)
        out[f"{tag}_fade_t"] = float(fade.mean() / fade.std() * np.sqrt(len(fade))) if len(fade) > 5 else float("nan")
    return out


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per UTC hour (in-session): basis intraday std, perp/home 5m std, beta, dev-reversion t."""
    sess = col(df, "sess")
    lp = lg(col(df, "perp"))
    lh = lg(col(df, "home_usd"))
    b = col(df, "basis").where(sess)
    day = pd.Series(days(dtidx(df)), index=df.index)
    b_intra = b - b.groupby(day).transform("mean")
    dev = b - _roll_mean(b, 78)
    fwd = lp.shift(-1) - lp
    rp, rh = lp.diff(), lh.diff()
    hours = np.asarray(cast(Any, dtidx(df)).hour)
    rows: List[Dict[str, float]] = []
    for h in sorted(set(hours[sess.to_numpy(dtype=bool)])):
        m = sess & (hours == h)
        bb, _, r2, n = ols(cast(pd.Series, rp[m]), cast(pd.Series, rh[m]))
        bd, td, _, _ = ols(fwd[m], dev[m])
        rows.append({"hour": float(h), "n": float(n), "basis_intra_std_bps": float(b_intra[m].std() * BPS),
                     "perp5m_std_bps": float(rp[m].std() * BPS), "home5m_std_bps": float(rh[m].std() * BPS),
                     "beta5m": bb, "r2": r2, "fwd_on_dev_b": bd, "fwd_on_dev_t": td})
    return pd.DataFrame(rows).set_index("hour")


def lag_trade(df: pd.DataFrame, thr_sd: float = 1.0) -> Dict[str, float]:
    """Perp-only lag trade: when the home line moved more than the perp over the last 5m
    (|home5m - perp5m| > thr_sd * its std), trade the perp in that direction; report the
    mean perp move over the next 5m / 30m (bps, one leg, no costs)."""
    sess = col(df, "sess")
    lp = lg(col(df, "perp"))
    lh = lg(col(df, "home_usd"))
    sig = (lh.diff() - lp.diff()).where(sess)
    sd = float(sig.std())
    day = pd.Series(days(dtidx(df)), index=df.index)
    out: Dict[str, float] = {"sig_sd_bps": sd * BPS}
    for k, tag in [(1, "5m"), (6, "30m")]:
        fwd = lp.shift(-k) - lp
        ok = sess & sess.shift(-k, fill_value=False) & (day == day.shift(-k)) & (sig.abs() > thr_sd * sd)
        pnl = (np.sign(sig[ok]) * fwd[ok]).dropna()
        out[f"n_{tag}"] = float(len(pnl))
        out[f"mean_{tag}_bps"] = float(pnl.mean() * BPS)
        out[f"t_{tag}"] = float(pnl.mean() / pnl.std() * np.sqrt(len(pnl))) if len(pnl) > 5 else float("nan")
        out[f"win_{tag}_pct"] = float((pnl > 0).mean() * 100)
    n_days = float(days(dtidx(sess[sess])).nunique())
    out["per_day"] = out["n_5m"] / max(n_days, 1)
    return out


def run_pair(p: Pair) -> Dict[str, object]:
    df = build_frame(p)
    res: Dict[str, object] = {"pair": p.name, "home_desc": p.home_desc}
    res["basis"] = basis_stats(df)
    res["beta"] = betas(df)
    res["leadlag"] = lead_lag(df)
    res["pred"] = predictive(df)
    res["mr_k2"] = mr_trade(df, 2.0, 0.5)
    res["mr_k3"] = mr_trade(df, 3.0, 0.5)
    res["snap"] = snap(df, p)
    res["lag1"] = lag_trade(df, 1.0)
    res["lag2"] = lag_trade(df, 2.0)
    res["hourly"] = hourly_profile(df)
    return res


def fmt(res: Dict[str, object]) -> str:
    lines = [f"\n######## {res['pair']}  ({res['home_desc']})"]
    for key in ["basis", "beta", "pred", "mr_k2", "mr_k3", "snap", "lag1", "lag2"]:
        d = res[key]
        assert isinstance(d, dict)
        lines.append(f"-- {key}")
        lines.append("   " + ", ".join(f"{k}={v:.3g}" for k, v in d.items()))
    ll = res["leadlag"]
    assert isinstance(ll, pd.Series)
    lines.append("-- leadlag corr(perp_t, home_{t-k}) k=-6..6 (k>0 home leads):")
    lines.append("   " + " ".join(f"{k:+d}:{v:.3f}" for k, v in ll.items()))
    hp = res["hourly"]
    assert isinstance(hp, pd.DataFrame)
    lines.append("-- hourly profile (UTC hour)")
    lines.append(hp.round(3).to_string())
    return "\n".join(lines)


def main(argv: List[str]) -> None:
    sel = argv[1:]
    for p in PAIRS_ADR:
        if sel and p.name not in sel and p.perp not in sel:
            continue
        try:
            res = run_pair(p)
        except FileNotFoundError as e:
            print(f"\n######## {p.name}: missing data {e}")
            continue
        print(fmt(res))


if __name__ == "__main__":
    main(sys.argv)
