#!/usr/bin/env python3
"""Session-structure analysis for Binance commodity perps vs CME futures.

(a) perp move during CME-closed windows vs future gap / first-30-min return
    after reopen; (d) extreme 5m moves in closed windows -> forward 60/180m;
(e) largest weekend moves and reopen behaviour. Prints markdown tables.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd

from scan_cmdty_common import (BN_DIR, MAP, NY, cme_session, col, dollar_depth, fmt, load_bn,
                               load_ib, rows)


def _win_ret(px: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    """Return px(t1)/px(t0)-1 using last available bar close at/before each ts."""
    a = px.loc[:t0]
    b = px.loc[:t1]
    if a.empty or b.empty or a.index[-1] < t0 - pd.Timedelta("30min") \
            or b.index[-1] < t1 - pd.Timedelta("30min"):
        return float("nan")
    return float(b.iloc[-1] / a.iloc[-1] - 1)


def closed_windows(idx: pd.DatetimeIndex) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """(kind, close_ts, reopen_ts) in UTC for each daily break / weekend in idx range."""
    out: List[Tuple[str, pd.Timestamp, pd.Timestamp]] = []
    d0 = cast(pd.Timestamp, idx[0]).tz_convert(NY).normalize()
    d1 = cast(pd.Timestamp, idx[-1]).tz_convert(NY).normalize()
    for day in pd.date_range(d0, d1, freq="D"):
        c = (day + pd.Timedelta(hours=17)).tz_convert("UTC")
        if day.dayofweek == 4:
            out.append(("weekend", c, (day + pd.Timedelta(days=2, hours=18)).tz_convert("UTC")))
        elif day.dayofweek < 4:
            out.append(("break", c, (day + pd.Timedelta(hours=18)).tz_convert("UTC")))
    return [w for w in out if w[1] >= idx[0] and w[2] <= idx[-1]]


def reopen_stats(sym: str) -> Dict[str, Any]:
    perp = col(load_bn(sym), "c")
    ibk = MAP[sym]["ib"]
    fut = load_ib(ibk) if ibk else None
    recs: List[Dict[str, Any]] = []
    for kind, c, r in closed_windows(pd.DatetimeIndex(perp.index)):
        r0 = cast(pd.Timestamp, r - pd.Timedelta("5min"))
        perp_closed = _win_ret(perp, c, r0)
        perp_post60 = _win_ret(perp, r0, cast(pd.Timestamp, r + pd.Timedelta("55min")))
        perp_post180 = _win_ret(perp, r0, cast(pd.Timestamp, r + pd.Timedelta("175min")))
        row: Dict[str, Any] = {"kind": kind, "close": c, "perp_closed": perp_closed,
                                  "perp_post60": perp_post60, "perp_post180": perp_post180}
        if fut is not None:
            fc = fut.loc[:c - pd.Timedelta("5min")]
            fo = fut.loc[r:r + pd.Timedelta("30min")]
            if not fc.empty and not fo.empty and fc.index[-1] >= c - pd.Timedelta("60min"):
                gap = float(fo["o"].iloc[0] / fc["c"].iloc[-1] - 1)
                f30 = float(fo["c"].iloc[min(5, len(fo) - 1)] / fo["o"].iloc[0] - 1)
                row.update({"fut_gap": gap, "fut_first30": f30})
        recs.append(row)
    df = pd.DataFrame(recs)
    out: Dict[str, Any] = {"sym": sym}
    for kind in ("break", "weekend"):
        d = rows(df, col(df, "kind") == kind).dropna(subset=["perp_closed"])
        pc = col(d, "perp_closed")
        k: Dict[str, Any] = {"n": int(len(d)),
                             "perp_closed_std_bps": float(pc.std() * 1e4),
                             "perp_closed_absmean_bps": float(pc.abs().mean() * 1e4)}
        if len(d) > 5:
            for cn in ("perp_post60", "perp_post180"):
                dd = d.dropna(subset=[cn])
                x, y = col(dd, "perp_closed"), col(dd, cn)
                k[f"beta_{cn}"] = float(np.polyfit(x, y, 1)[0]) if len(dd) > 5 else float("nan")
                k[f"corr_{cn}"] = float(x.corr(y))
        if "fut_gap" in d.columns:
            dd = d.dropna(subset=["fut_gap", "fut_first30"])
            k["n_fut"] = int(len(dd))
            if len(dd) > 5:
                x, g, f30 = col(dd, "perp_closed"), col(dd, "fut_gap"), col(dd, "fut_first30")
                bg = np.polyfit(x, g, 1)
                k["beta_gap"] = float(bg[0])
                k["corr_gap"] = float(x.corr(g))
                k["resid_gap_std_bps"] = float((g - np.polyval(bg, x)).std() * 1e4)
                k["beta_first30"] = float(np.polyfit(x, f30, 1)[0])
                k["corr_first30"] = float(x.corr(f30))
                k["fut_gap_std_bps"] = float(g.std() * 1e4)
        out[kind] = k
    out["_df"] = df
    return out


def extreme_moves(sym: str, q: float = 0.995) -> Dict[str, Any]:
    px = load_bn(sym)
    lp = cast(pd.Series, np.log(col(px, "c")))
    r = lp.diff()
    sess = cme_session(pd.DatetimeIndex(px.index))
    fwd60 = lp.shift(-12) - lp
    fwd180 = lp.shift(-36) - lp
    thr = float(r.abs().quantile(q))
    out: Dict[str, Any] = {"sym": sym, "thr_bps": thr * 1e4,
                              "sigma5m_open_bps": float(r[sess == "open"].std() * 1e4),
                              "sigma5m_break_bps": float(r[sess == "break"].std() * 1e4),
                              "sigma5m_weekend_bps": float(r[sess == "weekend"].std() * 1e4)}
    for kind in ("open", "break", "weekend", "closed"):
        m = (r.abs() > thr) & ((sess != "open") if kind == "closed" else (sess == kind))
        sgn = np.sign(r[m])
        c60 = (fwd60[m] * sgn).dropna()
        c180 = (fwd180[m] * sgn).dropna()
        out[kind] = {"n": int(m.sum()), "n_bars": int((sess == kind).sum()) if kind != "closed" else int((sess != "open").sum()),
                     "cont60_bps": float(c60.mean() * 1e4), "cont60_hit": float((c60 > 0).mean()) if len(c60) else np.nan,
                     "cont180_bps": float(c180.mean() * 1e4), "cont180_hit": float((c180 > 0).mean()) if len(c180) else np.nan,
                     "t60": float(c60.mean() / c60.std() * np.sqrt(len(c60))) if len(c60) > 2 else np.nan}
    return out


def depth_table() -> pd.DataFrame:
    files = sorted(BN_DIR.glob("depth_*.json"))
    snap = json.loads(files[-1].read_text())
    rows = []
    for s, book in snap["books"].items():
        d10, d25, d50 = (dollar_depth(book, b) for b in (10, 25, 50))
        rows.append({"sym": s, "spread_bps": d10["spread_bps"],
                     "bid10k": d10["bid"] / 1e3, "ask10k": d10["ask"] / 1e3,
                     "bid25k": d25["bid"] / 1e3, "ask25k": d25["ask"] / 1e3,
                     "bid50k": d50["bid"] / 1e3, "ask50k": d50["ask"] / 1e3})
    df = pd.DataFrame(rows)
    df.attrs["ts"] = pd.Timestamp(snap["ts"], unit="ms", tz="UTC")
    return df


def largest_weekends(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    w = rows(df, col(df, "kind") == "weekend").copy()
    w["abs"] = col(w, "perp_closed").abs()
    cols = [c for c in ("close", "perp_closed", "fut_gap", "fut_first30", "perp_post60",
                        "perp_post180") if c in w.columns]
    top = cast(pd.DataFrame, w.sort_values("abs", ascending=False).head(n)[cols])
    for c in cols[1:]:
        top[c] = (col(top, c) * 1e4).round(1)
    top["close"] = pd.to_datetime(col(top, "close")).dt.strftime("%Y-%m-%d")
    return top


def main() -> None:
    print("## (a) CME 闭市窗口：perp 移动 vs 期货重开缺口/首 30 分钟\n")
    print("| perp | 窗口 | N | perp闭市σ(bps) | N_fut | β(gap~perp) | corr | resid σ | β(first30~perp) | corr30 | perp post60 β | perp post180 β |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    saved: Dict[str, pd.DataFrame] = {}
    for sym in MAP:
        st = reopen_stats(sym)
        saved[sym] = cast(pd.DataFrame, st.pop("_df"))
        for kind in ("break", "weekend"):
            k = st[kind]
            assert isinstance(k, dict)
            g = lambda key: fmt(float(k.get(key, np.nan)), 2)  # noqa: E731
            print(f"| {sym} | {kind} | {k['n']} | {fmt(float(k['perp_closed_std_bps']))} | {k.get('n_fut','-')} | "
                  f"{g('beta_gap')} | {g('corr_gap')} | {fmt(float(k.get('resid_gap_std_bps', np.nan)))} | "
                  f"{g('beta_first30')} | {g('corr_first30')} | {g('beta_perp_post60')} | {g('beta_perp_post180')} |")
    print("\n## (e) 最大周末移动 (bps)\n")
    for sym in ("XAUUSDT", "XAGUSDT", "CLUSDT", "BZUSDT", "NATGASUSDT"):
        print(f"\n### {sym}\n")
        print(largest_weekends(saved[sym]).to_markdown(index=False))
    print("\n## (d) 极端 5m 移动 (|r| > 全样本 99.5 分位) 之后 60/180 分钟同向收益\n")
    print("| perp | thr bps | σ5m open/break/wknd | 窗口 | N | cont60 bps | hit60 | t60 | cont180 bps | hit180 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for sym in MAP:
        e = extreme_moves(sym)
        for kind in ("open", "break", "weekend"):
            k = e[kind]
            assert isinstance(k, dict)
            print(f"| {sym} | {fmt(float(e['thr_bps']))} | {fmt(float(e['sigma5m_open_bps']))}/{fmt(float(e['sigma5m_break_bps']))}/{fmt(float(e['sigma5m_weekend_bps']))} | {kind} | {k['n']} | "
                  f"{fmt(float(k['cont60_bps']))} | {fmt(float(k['cont60_hit']),2)} | {fmt(float(k['t60']))} | {fmt(float(k['cont180_bps']))} | {fmt(float(k['cont180_hit']),2)} |")
    d = depth_table()
    print(f"\n## (d2) 订单簿深度快照 {d.attrs['ts']} (k$)\n")
    print(d.round(1).to_markdown(index=False))


if __name__ == "__main__":
    main()
