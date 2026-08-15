#!/usr/bin/env python3
"""Q3/Q4: home-listed perps (KRX/HK) trading alone while the home market is closed.

Per pair:
  * off-hours perp move (home close -> next home open) vs home open gap
  * perp own reversal at home open (00:00-00:30 UTC KRX / 01:30-02:00 HK)
    vs prior 8h perp move
  * post-open home drift on the un-priced part of the perp move
  * SAMSUNG perp vs EWY perp during US hours vs KRX open gap
  * tail: open-print outliers in the home line, FX 5m gaps
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd

from scan_adr_basis import basis_stats, betas, lead_lag, mr_trade
from scan_adr_common import PAIRS_HOME, Pair, build_frame, col, days, dtidx, lg, load_bn, load_ib, ols

BPS = 1e4


def _daily_marks(df: pd.DataFrame, p: Pair) -> pd.DataFrame:
    """Per home trading day: home prev close, home open (first bar open & close),
    perp at prev close, perp just before open, perp 30m after open, etc."""
    sess = col(df, "sess")
    d_home = df.loc[sess & col(df, "home_raw").notna()]
    day = days(dtidx(d_home))
    home_first = d_home.groupby(day).head(1)
    home_last = d_home.groupby(day).tail(1)
    home_open_ts = pd.Series(dtidx(home_first), index=days(dtidx(home_first)))
    home_close_ts = pd.Series(dtidx(home_last), index=days(dtidx(home_last)))
    open_min = p.session[0][0]
    rows: List[Dict[str, float]] = []
    lp = lg(col(df, "perp"))
    lh = lg(col(df, "home_usd"))
    dlist = sorted(set(home_open_ts.index) & set(home_close_ts.index))
    for i in range(1, len(dlist)):
        d0, d1 = dlist[i - 1], dlist[i]
        t_close = pd.Timestamp(cast(Any, home_close_ts[d0]))
        t_open = pd.Timestamp(cast(Any, home_open_ts[d1]))
        if (t_open - t_close).total_seconds() > 4 * 86400:
            continue
        # sanity: open bar must be the session's first slot
        if (t_open.hour * 60 + t_open.minute) != open_min:
            continue
        pre = t_open - pd.Timedelta(minutes=5)   # perp bar closing exactly at open
        h8 = t_open - pd.Timedelta(hours=8)
        post30 = t_open + pd.Timedelta(minutes=25)
        post60 = t_open + pd.Timedelta(minutes=55)
        us0 = (t_open - pd.Timedelta(days=1)).replace(hour=13, minute=25)
        us1 = (t_open - pd.Timedelta(days=1)).replace(hour=19, minute=55)
        try:
            r = {
                "day": d1,
                "perp_off": lp[pre] - lp[t_close],
                "perp_prior8h": lp[pre] - lp[h8],
                "perp_us": lp[us1] - lp[us0],
                "perp_post30": lp[post30] - lp[pre],
                "perp_post60": lp[post60] - lp[pre],
                "home_gap_open": lh[t_open] - lh[t_close],   # USD terms, close->first bar close
                "home_post30": lh[post30] - lh[t_open],
                "home_post60": lh[post60] - lh[t_open],
                "basis_pre": lp[pre] - lh[t_open],       # perp vs first home print
                "basis_open5": lp[t_open] - lh[t_open],  # both known at open+5m (no look-ahead)
                "perp_post30b": lp[post30] - lp[t_open],
                "perp_post60b": lp[post60] - lp[t_open],
                "basis_close": lp[t_close] - lh[t_close],
            }
        except KeyError:
            continue
        if any(np.isnan(v) for k, v in r.items() if k != "day"):
            continue
        rows.append(r)
    out = pd.DataFrame(rows).set_index("day")
    return out


def offhours_tests(m: pd.DataFrame) -> Dict[str, float]:
    o: Dict[str, float] = {"n_days": float(len(m))}
    b, t, r2, _ = ols(m["home_gap_open"], m["perp_off"])
    o.update({"gap_on_perpoff_b": b, "gap_on_perpoff_t": t, "gap_on_perpoff_r2": r2})
    o["gap_std_bps"] = float(m["home_gap_open"].std() * BPS)
    o["perpoff_std_bps"] = float(m["perp_off"].std() * BPS)
    o["resid_std_bps"] = float((m["home_gap_open"] - m["perp_off"]).std() * BPS)
    o["basis_pre_mean_bps"] = float(m["basis_pre"].mean() * BPS)
    o["basis_pre_std_bps"] = float(m["basis_pre"].std() * BPS)
    # perp reversal at open vs prior 8h / off-hours
    for x in ["perp_prior8h", "perp_off", "basis_pre"]:
        for y in ["perp_post30", "perp_post60"]:
            b, t, r2, _ = ols(m[y], m[x])
            o[f"{y}_on_{x}_b"] = b
            o[f"{y}_on_{x}_t"] = t
    for y in ["perp_post30b", "perp_post60b"]:   # no look-ahead version
        b, t, r2, _ = ols(m[y], m["basis_open5"])
        o[f"{y}_on_basis_open5_b"] = b
        o[f"{y}_on_basis_open5_t"] = t
    o["fade5_post30_mean_bps"] = float((-np.sign(m["basis_open5"]) * m["perp_post30b"]).mean() * BPS)
    o["fade5_post30_win_pct"] = float(((-np.sign(m["basis_open5"]) * m["perp_post30b"]) > 0).mean() * 100)
    o["basis_open5_std_bps"] = float(m["basis_open5"].std() * BPS)
    # home post-open drift on the un-priced part of the perp move (perp - gap = basis_pre)
    for y in ["home_post30", "home_post60"]:
        b, t, r2, _ = ols(m[y], m["basis_pre"])
        o[f"{y}_on_basis_pre_b"] = b
        o[f"{y}_on_basis_pre_t"] = t
    # US-session-only part of the perp move
    b, t, r2, _ = ols(m["home_gap_open"], m["perp_us"])
    o.update({"gap_on_perpUS_b": b, "gap_on_perpUS_t": t, "gap_on_perpUS_r2": r2})
    # fade trade: short perp at open if basis_pre>0
    o["fade_post30_mean_bps"] = float((-np.sign(m["basis_pre"]) * m["perp_post30"]).mean() * BPS)
    o["fade_post30_win_pct"] = float(((-np.sign(m["basis_pre"]) * m["perp_post30"]) > 0).mean() * 100)
    big = m["basis_pre"].abs() > m["basis_pre"].abs().median()
    o["fade_post30_bigHalf_mean_bps"] = float(
        (-np.sign(m.loc[big, "basis_pre"]) * m.loc[big, "perp_post30"]).mean() * BPS)
    return o


def samsung_vs_ewy(m: pd.DataFrame) -> Dict[str, float]:
    """Compare SAMSUNG perp off-hours move with EWY perp over the same window."""
    ewy = lg(col(load_bn("EWY"), "c"))
    krw = lg(col(load_ib("USDKRW"), "c"))
    o: Dict[str, float] = {}
    rows = []
    for d in m.index:
        t_close = (d - pd.Timedelta(days=1)).replace(hour=6, minute=25) if d.dayofweek else \
            (d - pd.Timedelta(days=3)).replace(hour=6, minute=25)
        pre = d.replace(hour=23, minute=55) - pd.Timedelta(days=1)
        us0 = (d - pd.Timedelta(days=1)).replace(hour=13, minute=25)
        us1 = (d - pd.Timedelta(days=1)).replace(hour=19, minute=55)
        try:
            k0 = krw.asof(t_close)
            k1 = krw.asof(pre)
            rows.append({"day": d, "ewy_off": ewy[pre] - ewy[t_close] + (k1 - k0),
                         "ewy_us": ewy[us1] - ewy[us0]})
        except KeyError:
            continue
    e = pd.DataFrame(rows).set_index("day")
    mm = m.join(e, how="inner").dropna()
    o["n"] = float(len(mm))
    b, t, r2, _ = ols(mm["home_gap_open"], mm["ewy_off"])
    o.update({"gap_on_ewyoff_b": b, "gap_on_ewyoff_t": t, "gap_on_ewyoff_r2": r2})
    b, t, r2, _ = ols(mm["home_gap_open"], mm["ewy_us"])
    o.update({"gap_on_ewyUS_b": b, "gap_on_ewyUS_t": t, "gap_on_ewyUS_r2": r2})
    b, t, r2, _ = ols(mm["perp_off"], mm["ewy_off"])
    o.update({"perpoff_on_ewyoff_b": b, "perpoff_on_ewyoff_t": t, "perpoff_on_ewyoff_r2": r2})
    # multivariate-ish: gap on perp_off after controlling ewy_off (residualise)
    b1, _, _, _ = ols(mm["perp_off"], mm["ewy_off"])
    resid = mm["perp_off"] - b1 * mm["ewy_off"]
    b, t, r2, _ = ols(mm["home_gap_open"] - o["gap_on_ewyoff_b"] * mm["ewy_off"], resid)
    o.update({"gap_on_perpoff_resid_b": b, "gap_on_perpoff_resid_t": t})
    return o


def tail_checks(df: pd.DataFrame, p: Pair) -> Dict[str, float]:
    """Home-line open-print outliers & FX gaps."""
    o: Dict[str, float] = {}
    sess = col(df, "sess")
    hr = col(df.loc[sess & col(df, "home_raw").notna()], "home_raw")
    day = days(dtidx(hr))
    first = hr.groupby(day).first()
    second = hr.groupby(day).nth(1)
    second.index = days(dtidx(second))
    prev_close = hr.groupby(day).last().shift(1)
    gap = np.log(first / prev_close).dropna()
    rev = np.log(second / first).dropna()
    o["open_gap_std_bps"] = float(gap.std() * BPS)
    o["open_gap_maxabs_bps"] = float(gap.abs().max() * BPS)
    o["first5m_rev_std_bps"] = float(rev.std() * BPS)
    o["first5m_rev_maxabs_bps"] = float(rev.abs().max() * BPS)
    o["days_first5m_rev_gt100bps"] = float((rev.abs() > 100e-4).sum())
    b = col(df.loc[sess], "basis").dropna()
    fb = b.groupby(days(dtidx(b))).first()
    o["basis_firstbar_std_bps"] = float(fb.std() * BPS)
    o["basis_firstbar_maxabs_bps"] = float(fb.abs().max() * BPS)
    o["basis_maxabs_bps"] = float(b.abs().max() * BPS)
    o["basis_gt200bps_bars"] = float((b.abs() > 200e-4).sum())
    if p.fx:
        fx = lg(col(load_ib(p.fx), "c")).diff().dropna()
        o["fx_5m_std_bps"] = float(fx.std() * BPS)
        o["fx_5m_maxabs_bps"] = float(fx.abs().max() * BPS)
        o["fx_gt30bps_5m_bars"] = float((fx.abs() > 30e-4).sum())
    return o


def run_pair(p: Pair) -> Tuple[str, Dict[str, Dict[str, float]]]:
    df = build_frame(p)
    m = _daily_marks(df, p)
    res: Dict[str, Dict[str, float]] = {
        "basis_in_session": basis_stats(df),
        "beta_in_session": betas(df),
        "mr_k2_in_session": mr_trade(df, 2.0, 0.5),
        "offhours": offhours_tests(m),
        "tail": tail_checks(df, p),
    }
    if p.perp in ("SAMSUNG", "SKHYNIX", "HYUNDAI"):
        res["vs_ewy"] = samsung_vs_ewy(m)
    ll = lead_lag(df)
    res["leadlag"] = {str(k): v for k, v in ll.items()}
    return p.name, res


def main(argv: List[str]) -> None:
    sel = argv[1:]
    for p in PAIRS_HOME:
        if sel and p.perp not in sel:
            continue
        try:
            name, res = run_pair(p)
        except FileNotFoundError as e:
            print(f"\n######## {p.name}: missing {e}")
            continue
        print(f"\n######## {name}")
        for k, d in res.items():
            print(f"-- {k}")
            print("   " + ", ".join(f"{kk}={v:.3g}" for kk, v in d.items()))


if __name__ == "__main__":
    main(sys.argv)
