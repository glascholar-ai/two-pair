#!/usr/bin/env python3
"""Q1: basis level / dispersion / half-life by segment + threshold trade sim.
Q2: open snap at 08:00 / 13:30 UTC (perp vs IBKR anchor regressions).
Q4: ex-dividend behaviour around Binance 'Special' funding events.

Usage: python3 scan_basis_stats.py [SYM ...]  (default: all cached symbols)
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd

from scan_basis_prep import IBD, SEG_ORDER, load_all, load_special_funding, minute_of_day

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# round-trip cost assumptions (bps of notional per leg pair)
FEE_PERP_TAKER = 4.0        # Binance taker, per side
FEE_IB = 1.5                # IBKR per side (commission)


def md(df: pd.DataFrame) -> str:
    return str(df.to_markdown())


def mean_row(df: pd.DataFrame, name: str, nd: int) -> str:
    """One-row markdown table of column means."""
    return md(pd.DataFrame({name: cast(pd.Series, df.mean(numeric_only=True))}).T.round(nd))


def arr(x: object) -> np.ndarray:
    return np.asarray(cast(Any, x), dtype=float)


# ------------------------------------------------------------------ Q1 -----
def seg_stats(m: pd.DataFrame, col: str) -> pd.DataFrame:
    d = cast(pd.DataFrame, m[[col, "seg"]]).dropna()
    g = d.groupby("seg")[col]
    out = pd.DataFrame({"mean": g.mean(), "median": g.median(), "std": g.std(),
                        "mean_abs": g.apply(lambda x: float(np.abs(arr(x)).mean())),
                        "p95_abs": g.apply(lambda x: float(np.quantile(np.abs(arr(x)), 0.95))),
                        "n": g.count()})
    return out.reindex(SEG_ORDER)


def half_life(m: pd.DataFrame, col: str) -> pd.DataFrame:
    """AR(1) on 5m basis within contiguous segment runs (per trading day) -> phi, HL (min)."""
    rows: Dict[str, Dict[str, float]] = {}
    for seg in SEG_ORDER:
        x_all: List[np.ndarray] = []
        y_all: List[np.ndarray] = []
        sub = cast(pd.DataFrame, m[m["seg"] == seg])
        for _, d in sub.groupby("tday"):
            b = arr(cast(pd.Series, d[col]).dropna().values)
            if len(b) < 6:
                continue
            x_all.append(b[:-1])
            y_all.append(b[1:])
        if not x_all:
            continue
        x = np.concatenate(x_all)
        y = np.concatenate(y_all)
        xc, yc = x - x.mean(), y - y.mean()
        phi = float((xc * yc).sum() / (xc * xc).sum())
        hl = float(-5 * np.log(2) / np.log(phi)) if 0 < phi < 1 else float("inf")
        rows[seg] = {"phi_5m": phi, "half_life_min": hl, "n": float(len(x))}
    return pd.DataFrame(rows).T.reindex(SEG_ORDER)


def spread_stats(m: pd.DataFrame) -> pd.DataFrame:
    g = m.groupby("seg")["spread_bps"]
    return pd.DataFrame({"spread_med": g.median(), "spread_mean": g.mean(),
                         "spread_p90": g.quantile(0.9), "n": g.count()}).reindex(SEG_ORDER)


def trade_sim(m: pd.DataFrame, col: str, thr: float, exit_bps: float = 10.0,
              max_hold_bars: int = 288, entry_segs: Tuple[str, ...] = ("AH", "PRE"),
              ) -> pd.DataFrame:
    """Enter when |basis|>thr inside entry_segs, exit when |basis|<exit_bps or timeout.

    Gross pnl (bps) = perp leg + stock leg (log price changes, sign = fade the basis).
    Executed at the *next* bar close after the signal bar. One position at a time.
    Stock leg priced off ib_midc when col is a mid-based basis, else ib_last.
    """
    b = arr(m[col].values)
    seg = np.asarray(m["seg"].values)
    lb = np.log(arr(m["bn"].values))
    stk_col = "ib_midc" if col.startswith("basis_mid") else "ib_last"
    ls = np.log(arr(m[stk_col].values))
    ts = m.index
    n = len(m)
    rows: List[Dict[str, object]] = []
    i = 0
    while i < n - 2:
        ok = seg[i] in entry_segs and np.isfinite(b[i]) and abs(b[i]) > thr
        if ok and np.isfinite(b[i + 1]) and np.isfinite(ls[i + 1]):
            e = i + 1
            sgn = -np.sign(b[e])           # perp rich -> short perp / long stock
            j = e + 1
            while j < n and j - e < max_hold_bars:
                if np.isfinite(b[j]) and abs(b[j]) < exit_bps:
                    break
                j += 1
            j = min(j, n - 1)
            while j > e and not np.isfinite(ls[j]):
                j -= 1
            pnl_perp = sgn * (lb[j] - lb[e]) * 1e4
            pnl_stk = -sgn * (ls[j] - ls[e]) * 1e4
            rows.append({"t_in": ts[e], "seg_in": seg[e], "basis_in": b[e], "basis_out": b[j],
                         "hold_min": 5 * (j - e), "gross": pnl_perp + pnl_stk,
                         "perp_leg": pnl_perp, "stock_leg": pnl_stk,
                         "converged": bool(np.isfinite(b[j]) and abs(b[j]) < exit_bps)})
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(rows)


def sim_table(data: Dict[str, pd.DataFrame], col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    for thr in (30, 50, 100):
        for s, m in data.items():
            mm = cast(pd.DataFrame, m.dropna(subset=[col]))
            if mm.empty:
                continue
            tr = trade_sim(mm, col, thr)
            ndays = mm["tday"].nunique()
            spread = float(cast(pd.Series, m.loc[m["seg"].isin(["AH", "PRE"]), "spread_bps"]).mean())
            if tr.empty:
                rows.append({"col": col, "thr": thr, "sym": s, "n": 0, "per_day": 0.0})
                continue
            gross = float(tr["gross"].mean())
            cost = 2 * FEE_PERP_TAKER + 2 * FEE_IB + spread
            rows.append({"col": col, "thr": thr, "sym": s, "n": len(tr),
                         "per_day": len(tr) / ndays, "gross_mean": gross,
                         "gross_med": float(tr["gross"].median()),
                         "perp_leg": float(tr["perp_leg"].mean()),
                         "stock_leg": float(tr["stock_leg"].mean()),
                         "cost_est": cost, "net_mean": gross - cost,
                         "hold_med_min": float(tr["hold_min"].median()),
                         "conv%": float(tr["converged"].mean() * 100),
                         "AH%": float((tr["seg_in"] == "AH").mean() * 100)})
    df = pd.DataFrame(rows)
    aggs: List[Dict[str, object]] = []
    for thr, d in df.groupby("thr"):
        nn = arr(d["n"].values)
        w = nn / nn.sum() if nn.sum() else nn
        aggs.append({"thr": thr, "n": int(nn.sum()), "per_sym_day": float(d["per_day"].mean()),
                     "gross_wmean": float(np.nansum(arr(d.get("gross_mean", np.nan)) * w)),
                     "net_wmean": float(np.nansum(arr(d.get("net_mean", np.nan)) * w)),
                     "perp_leg": float(np.nansum(arr(d.get("perp_leg", np.nan)) * w)),
                     "stock_leg": float(np.nansum(arr(d.get("stock_leg", np.nan)) * w))})
    return df, pd.DataFrame(aggs).set_index("thr")


def q1(data: Dict[str, pd.DataFrame]) -> str:
    out: List[str] = ["## Q1 基差水平 / 离散度 / 半衰期\n"]
    per_sym = []
    for s, m in data.items():
        st = seg_stats(m, "basis")
        st["sym"] = s
        per_sym.append(st)
    allst = pd.concat(per_sym)
    out.append("### 各标的按时段的基差(bps, ln(perp/IB last)) 均值 / |基差|均值 / std\n")
    piv = allst.reset_index().pivot(index="sym", columns="seg", values=["mean", "mean_abs", "std"])
    piv = piv.reindex(columns=pd.MultiIndex.from_product([["mean", "mean_abs", "std"], SEG_ORDER]))
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    out.append(md(piv.round(1)) + "\n")
    pooled = pd.concat([m.assign(sym=s) for s, m in data.items()])
    out.append("### 全样本汇总 (basis vs IB last, 仅有成交的 bar)\n")
    out.append(md(seg_stats(pooled, "basis").round(2)) + "\n")
    out.append("### 全样本汇总 (basis vs IB 时间平均 mid, BID_ASK 覆盖期 ~30 个交易日)\n")
    out.append(md(seg_stats(pooled, "basis_mid").round(2)) + "\n")
    out.append("### 全样本汇总 (basis vs 居中 mid = avg(mid_t, mid_t+1))\n")
    out.append(md(seg_stats(pooled, "basis_midc").round(2)) + "\n")
    out.append("### IBKR 买卖价差 (bps, 时间平均) 按时段\n")
    out.append(md(spread_stats(pooled).round(1)) + "\n")
    for col in ("basis", "basis_midc"):
        out.append(f"### 基差 AR(1) 半衰期 (5m, 段内, 全样本, {col})\n")
        out.append(md(half_life(pooled, col).round(3)) + "\n")
    hl_sym = {s: half_life(m, "basis")["half_life_min"] for s, m in data.items()}
    out.append("### 各标的半衰期(分钟, basis vs last)\n")
    out.append(md(pd.DataFrame(hl_sym).T.round(0)) + "\n")
    out.append("### 阈值套利模拟 (进场 |basis|>thr 于 AH/PRE, 次 bar 执行, 出场 |basis|<10bps 或 24h)\n"
               f"成本估计 cost_est = perp taker {FEE_PERP_TAKER}×2 + IB {FEE_IB}×2 + AH/PRE 平均价差(一次全价差)\n")
    for col in ("basis", "basis_midc"):
        df, agg = sim_table(data, col)
        out.append(f"#### 信号 = {col}\n")
        out.append(str(df.round(1).to_markdown(index=False)) + "\n")
        out.append("汇总(按笔数加权):\n\n" + md(agg.round(1)) + "\n")
    return "\n".join(out)


# ------------------------------------------------------------------ Q2 -----
def _at(m: pd.DataFrame, col: str, hh: int, mm_: int) -> pd.Series:
    """Value of col at bar hh:mm keyed by tday."""
    mod = minute_of_day(pd.DatetimeIndex(m.index))
    sel = cast(pd.DataFrame, m[mod == hh * 60 + mm_])
    s = pd.Series(arr(sel[col].values), index=pd.DatetimeIndex(sel["tday"]))
    return cast(pd.Series, s[~s.index.duplicated()])


def _ols(y: pd.Series, x: pd.Series) -> Tuple[float, float, float, int]:
    d = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(d) < 8:
        return np.nan, np.nan, np.nan, len(d)
    yy, xx = arr(d["y"].values), arr(d["x"].values)
    X = np.column_stack([np.ones(len(xx)), xx])
    beta = np.linalg.lstsq(X, yy, rcond=None)[0]
    resid = yy - X @ beta
    s2 = float(resid @ resid) / (len(xx) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    t = float(beta[1] / np.sqrt(cov[1, 1]))
    r2 = 1 - float(resid @ resid) / float((yy - yy.mean()) @ (yy - yy.mean()))
    return float(beta[1]), t, r2, len(d)


def open_snap_table(m: pd.DataFrame) -> Dict[str, float]:
    """Per-symbol regressions around 08:00 and 13:30 UTC (all in bps, log)."""
    mm = m.assign(lb=np.log(arr(m["bn"].values)) * 1e4,
                  ls=np.log(arr(m["ib_last"].values)) * 1e4)
    ib_2355 = _at(mm, "ls", 23, 55)          # AH close, keyed to next tday
    pp_2355 = _at(mm, "lb", 23, 55)
    pp_0755 = _at(mm, "lb", 7, 55)
    pp_0805 = _at(mm, "lb", 8, 5)
    pp_0830 = _at(mm, "lb", 8, 30)
    ib_0800 = _at(mm, "ls", 8, 0)            # first premarket bar close (~08:05)
    ib_0745 = _at(mm, "ls", 7, 45)           # IBKR overnight last
    pp_1325 = _at(mm, "lb", 13, 25)
    pp_1335 = _at(mm, "lb", 13, 35)
    pp_1400 = _at(mm, "lb", 14, 0)
    ib_1325 = _at(mm, "ls", 13, 25)
    ib_1335 = _at(mm, "ls", 13, 35)
    res: Dict[str, float] = {}
    b, t, r2, n = _ols(ib_0800 - ib_2355, pp_0755 - pp_2355)
    res.update({"a_slope": b, "a_t": t, "a_r2": r2, "a_n": n})
    gap = pp_0755 - ib_2355
    b, t, r2, n = _ols(pp_0830 - pp_0755, gap)
    res.update({"b_slope": b, "b_t": t, "b_r2": r2, "b_n": n})
    b, t, _, _ = _ols(pp_0805 - pp_0755, gap)
    res.update({"b10_slope": b, "b10_t": t})
    gap2 = pp_0755 - ib_0745
    b, t, r2, n = _ols(pp_0830 - pp_0755, gap2)
    res.update({"c_slope": b, "c_t": t, "c_n": n})
    b, t, _, _ = _ols(ib_0800 - ib_0745, gap2)
    res.update({"c_stk_slope": b, "c_stk_t": t})
    gap3 = pp_1325 - ib_1325
    b, t, r2, n = _ols(pp_1335 - pp_1325, gap3)
    res.update({"d_slope": b, "d_t": t, "d_n": n})
    b, t, _, _ = _ols(ib_1335 - ib_1325, gap3)
    res.update({"d_stk_slope": b, "d_stk_t": t})
    b, t, _, _ = _ols(pp_1400 - pp_1325, gap3)
    res.update({"d30_slope": b, "d30_t": t})
    res["gap0755_mean_abs"] = float(np.nanmean(np.abs(arr(gap.values))))
    res["gap0755_ovn_mean_abs"] = float(np.nanmean(np.abs(arr(gap2.values)))) if gap2.notna().any() else np.nan
    res["gap1325_mean_abs"] = float(np.nanmean(np.abs(arr(gap3.values))))
    return res


def perp_move_profile(m: pd.DataFrame) -> pd.Series:
    """Mean |5m perp log return| (bps) by minute-of-day, near 08:00 / 13:30 vs baseline."""
    r = np.abs(np.diff(np.log(arr(m["bn"].values)), prepend=np.nan)) * 1e4
    mod = minute_of_day(pd.DatetimeIndex(m.index))
    prof = pd.Series(r, index=mod).groupby(level=0).mean()
    keys = {"07:45": 465, "07:50": 470, "07:55": 475, "08:00": 480, "08:05": 485, "08:10": 490,
            "13:20": 800, "13:25": 805, "13:30": 810, "13:35": 815, "13:40": 820}
    out: Dict[str, float] = {k: float(cast(Any, prof.get(v, np.nan))) for k, v in keys.items()}
    pi = arr(prof.index.values)
    pv = arr(prof.values)
    out["DEAD_avg"] = float(pv[(pi >= 0) & (pi < 480)].mean())
    out["PRE_avg"] = float(pv[(pi >= 495) & (pi < 800)].mean())
    out["AH_avg"] = float(pv[pi >= 1200].mean())
    return pd.Series(out)


def q2(data: Dict[str, pd.DataFrame]) -> str:
    out = ["## Q2 开盘 snap: 08:00 (盘前) 与 13:30 (RTH) 前后\n"]
    tab = pd.DataFrame({s: open_snap_table(m) for s, m in data.items()}).T
    out.append("### 回归 (bps 对 bps)\n"
               "- a: r_stock(IB 23:55→08:00bar) ~ r_perp(23:55→07:55): 斜率≈1 → perp 是价格发现\n"
               "- b: r_perp(07:55→08:30) ~ gap=perp07:55−IB23:55: 负 → perp 被拉回旧锚 (b10: 07:55→08:05)\n"
               "- c: r_perp(07:55→08:30) ~ gap2=perp07:55−IB overnight 07:45 "
               "(c_stk: 股票 08:00bar 相对 07:45 的变动 ~ gap2)\n"
               "- d: r_perp(13:25→13:35) ~ gap3=perp13:25−IB13:25 (d_stk: 股票同窗口 ~ gap3; d30: perp 13:25→14:00)\n")
    out.append(md(tab.round(2)) + "\n")
    out.append("均值:\n\n" + mean_row(tab, "mean", 2) + "\n")
    prof = pd.DataFrame({s: perp_move_profile(m) for s, m in data.items()}).T
    out.append("### perp 5m |收益| (bps) 在 08:00 / 13:30 附近 vs 时段平均\n")
    out.append(md(prof.round(1)) + "\n")
    out.append("全样本均值:\n\n" + mean_row(prof, "mean_abs_bps", 1) + "\n")
    return "\n".join(out)


# ------------------------------------------------------------------ Q4 -----
def _seg_median(m: pd.DataFrame, day: pd.Timestamp, seg: str, col: str = "basis") -> float:
    d = cast(pd.Series, m.loc[(m["tday"] == day) & (m["seg"] == seg), col]).dropna()
    return float(d.median()) if len(d) else np.nan


def _jump(m: pd.DataFrame, col: str, t0: pd.Timestamp) -> float:
    try:
        before = float(cast(Any, m.at[t0 - pd.Timedelta(minutes=5), col]))
        after = float(cast(Any, m.at[t0 + pd.Timedelta(minutes=5), col]))
    except KeyError:
        return np.nan
    return float(np.log(after / before) * 1e4)


def q4(data: Dict[str, pd.DataFrame]) -> str:
    out = ["## Q4 除息 (Binance Special funding = 股息结算, 发生在 ex-date 00:00 UTC)\n"]
    sp = load_special_funding()
    sp = cast(pd.DataFrame, sp[sp["sym"].isin(list(data.keys()))])
    rows: List[Dict[str, object]] = []
    for sym, t, rate in zip(sp["sym"], sp["t"], sp["rate"]):
        m = data[str(sym)]
        t0 = cast(pd.Timestamp, cast(pd.Timestamp, pd.Timestamp(cast(Any, t))).normalize())
        tdays = pd.DatetimeIndex(m["tday"].unique()).sort_values()
        prev = tdays[tdays < t0]
        nxt = tdays[tdays > t0]
        t_prev = cast(pd.Timestamp, prev[-1]) if len(prev) else t0
        t_prev2 = cast(pd.Timestamp, prev[-2]) if len(prev) > 1 else t0
        t_next = cast(pd.Timestamp, nxt[0]) if len(nxt) else t0
        typ_dead = _typical(m, "DEAD")
        typ_pre = _typical(m, "PRE")
        typ_ah = _typical(m, "AH")
        contiguous = (t0 - t_prev) == pd.Timedelta(days=1)
        rows.append({"sym": sym, "ex_date": t0.date(), "div_bps": -float(rate) * 1e4,
                     "perp_jump_0000": _jump(m, "bn", t0) if contiguous else np.nan,
                     "stock_jump_0000": _jump(m, "ib_last_ff", t0) if contiguous else np.nan,
                     "basis_AH_-1": _seg_median(m, t0, "AH"), "AH_typ": typ_ah,
                     "basis_DEAD_ex": _seg_median(m, t0, "DEAD"), "DEAD_typ": typ_dead,
                     "basis_PRE_ex": _seg_median(m, t0, "PRE"), "PRE_typ": typ_pre,
                     "basis_OPEN_ex": _seg_median(m, t0, "OPEN"),
                     "basis_REST_ex": _seg_median(m, t0, "REST"),
                     "basis_REST_-1": _seg_median(m, t_prev, "REST"),
                     "basis_REST_-2": _seg_median(m, t_prev2, "REST"),
                     "basis_REST_+1": _seg_median(m, t_next, "REST")})
    df = pd.DataFrame(rows)
    df["DEAD_dev"] = df["basis_DEAD_ex"] - df["DEAD_typ"]
    df["PRE_dev"] = df["basis_PRE_ex"] - df["PRE_typ"]
    df["AH-1_dev"] = df["basis_AH_-1"] - df["AH_typ"]
    out.append(str(df.round(1).to_markdown(index=False)) + "\n")
    if len(df):
        num = df.drop(columns=["sym", "ex_date"])
        out.append("均值:\n\n" + mean_row(num, "mean", 1) + "\n")
        big = cast(pd.DataFrame, num[num["div_bps"] >= 5])
        if len(big):
            out.append("仅 div ≥ 5bps 的事件均值:\n\n" + mean_row(big, "mean", 1) + "\n")
            for col in ("AH-1_dev", "DEAD_dev", "PRE_dev"):
                r = arr(big[col].values) / arr(big["div_bps"].values)
                out.append(f"{col} / div: 均值 {np.nanmean(r):.2f}, 中位 {np.nanmedian(r):.2f} "
                           f"(若 perp 提前/滞后除息, 对应值应≈-1)\n")
    return "\n".join(out)


def _typical(m: pd.DataFrame, seg: str) -> float:
    d = cast(pd.Series, m.loc[m["seg"] == seg, "basis"]).dropna()
    return float(d.median()) if len(d) else np.nan


def main(argv: List[str]) -> None:
    syms = argv[1:] or sorted(p.stem.replace("_5m_ext", "") for p in IBD.glob("*_5m_ext.parquet"))
    data = load_all(syms)
    print(f"loaded {list(data)}", file=sys.stderr)
    cov = pd.DataFrame({s: {"bars": len(m), "days": m["tday"].nunique(),
                            "first": str(m.index.min())[:10], "last": str(m.index.max())[:10],
                            "ib_bars": int(m["ib_last"].notna().sum()),
                            "mid_bars": int(m["basis_mid"].notna().sum()),
                            "ovn_bars": int((m["ib_src"] == "OVERNIGHT").sum())}
                        for s, m in data.items()}).T
    print("## 数据覆盖\n\n" + md(cov) + "\n")
    print(q1(data))
    print(q2(data))
    print(q4(data))


if __name__ == "__main__":
    main(sys.argv)
