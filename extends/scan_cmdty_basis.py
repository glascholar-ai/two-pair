#!/usr/bin/env python3
"""(b) perp-future basis + funding; (c) Binance vs Hyperliquid cross-exchange
spread; FX perps (HL xyz:EUR/GBP/JPY) vs IBKR spot. Prints markdown."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd

from scan_cmdty_common import (FX_HL, MAP, bn_funding, cme_session, col, fmt, fx_session,
                               half_life, hl_funding, ib_meta, load_bn, load_hl, load_ib, rows)


def dev_stats(dev: pd.Series, label: str) -> Dict[str, Any]:
    """Stats for a deviation series (in bps) sampled at 5m."""
    d = dev.dropna()
    out: Dict[str, Any] = {"label": label, "n": int(len(d)), "mean": float(d.mean()),
                              "std": float(d.std()), "p95abs": float(d.abs().quantile(0.95)),
                              "hl_bars": half_life(d),
                              "gt20": float((d.abs() > 20).mean()), "gt50": float((d.abs() > 50).mean())}
    # episodes: crossing 20 bps -> time until back within 10 bps (bars)
    ab = d.abs().values
    times: List[int] = []
    i = 0
    while i < len(ab):
        if ab[i] > 20:
            j = i
            while j < len(ab) and ab[j] > 10:
                j += 1
            times.append(j - i)
            i = j
        else:
            i += 1
    out["n_ep20"] = len(times)
    out["ep20_med_bars"] = float(np.median(times)) if times else float("nan")
    out["ep20_p90_bars"] = float(np.quantile(times, 0.9)) if times else float("nan")
    return out


def basis_row(sym: str) -> Optional[Dict[str, Any]]:
    ibk = MAP[sym]["ib"]
    if not ibk:
        return None
    perp = col(load_bn(sym), "c")
    fut = col(load_ib(ibk), "c")
    j = pd.concat({"p": perp, "f": fut}, axis=1).dropna()
    sess = cme_session(pd.DatetimeIndex(j.index))
    j = rows(j, sess == "open")
    raw = cast(pd.Series, (col(j, "p") / col(j, "f") - 1) * 1e4)
    meta = ib_meta(ibk)
    exp = pd.Timestamp(str(meta.get("expiry", "20260827"))).tz_localize("UTC")
    tdays = np.asarray((exp - pd.DatetimeIndex(j.index)).days, dtype=float)
    implied_carry = float(np.median(-np.asarray(raw) / 1e4 / np.maximum(tdays, 1) * 365))
    # deviation from carry-adjusted fair: use 1-day rolling median as fair basis
    fair = raw.rolling(288, min_periods=100).median()
    dev = raw - fair
    st = dev_stats(dev, sym)
    st.update({"raw_mean_bps": float(raw.mean()), "raw_std_bps": float(raw.std()),
               "implied_carry_pct": implied_carry * 100, "expiry": str(meta.get("localSymbol", ibk)),
               "raw_last_bps": float(raw.iloc[-1]), "raw_p5": float(raw.quantile(0.05)),
               "raw_p95": float(raw.quantile(0.95))})
    # funding: BN 4h rate vs basis dev at funding time
    fr = bn_funding(sym)
    st["fund_mean_ann_pct"] = float(fr.mean() * 6 * 365 * 100)
    st["fund_std_bps"] = float(fr.std() * 1e4)
    st["fund_pos"] = float((fr > 0).mean())
    st["fund_last30d_ann_pct"] = float(fr.iloc[-180:].mean() * 6 * 365 * 100)
    dev_at = dev.reindex(fr.index, method="nearest", tolerance=pd.Timedelta("10min"))
    m = dev_at.notna()
    st["corr_fund_dev"] = float(np.corrcoef(fr[m], dev_at[m])[0, 1]) if m.sum() > 20 else float("nan")
    return st


def print_basis() -> None:
    print("## (b) Binance perp vs CME 期货基差（仅 CME 交易时段, 5m）\n")
    print("| perp | 期货 | N | 原始基差均值 bps | 原始σ | p5/p95 | 隐含年化carry% | 偏离σ(bps, vs 1d滚动中位) | 半衰期(bars) | \\|dev\\|>20 | >50 | 20→10 事件数 | 中位/ p90 bars |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    fund_rows: List[Dict[str, Any]] = []
    for sym in MAP:
        st = basis_row(sym)
        if not st:
            continue
        print(f"| {sym} | {st['expiry']} | {st['n']} | {fmt(float(st['raw_mean_bps']))} | {fmt(float(st['raw_std_bps']))} | "
              f"{fmt(float(st['raw_p5']))}/{fmt(float(st['raw_p95']))} | {fmt(float(st['implied_carry_pct']),2)} | "
              f"{fmt(float(st['std']))} | {fmt(float(st['hl_bars']))} | {fmt(float(st['gt20'])*100)}% | {fmt(float(st['gt50'])*100)}% | "
              f"{st['n_ep20']} | {fmt(float(st['ep20_med_bars']))}/{fmt(float(st['ep20_p90_bars']))} |")
        fund_rows.append(st)
    print("\n### 资金费 (Binance 4h) 与基差偏离\n")
    print("| perp | 全样本年化% | 近30d年化% | 单期σ bps | 正占比 | corr(funding, dev) |")
    print("|---|---|---|---|---|---|")
    for st in fund_rows:
        print(f"| {st['label']} | {fmt(float(st['fund_mean_ann_pct']),2)} | {fmt(float(st['fund_last30d_ann_pct']),2)} | "
              f"{fmt(float(st['fund_std_bps']),2)} | {fmt(float(st['fund_pos']),2)} | {fmt(float(st['corr_fund_dev']),2)} |")
    print("\n### 无期货参照的品种资金费\n")
    print("| perp | 全样本年化% | 近30d年化% | 单期σ bps | 正占比 |")
    print("|---|---|---|---|---|")
    for sym in MAP:
        if MAP[sym]["ib"]:
            continue
        fr = bn_funding(sym)
        print(f"| {sym} | {fmt(float(fr.mean()*6*365*100),2)} | {fmt(float(fr.iloc[-180:].mean()*6*365*100),2)} | "
              f"{fmt(float(fr.std()*1e4),2)} | {fmt(float((fr>0).mean()),2)} |")


def print_cross() -> None:
    print("\n## (c) Binance vs Hyperliquid(xyz) 同品种价差 ln(BN/HL) bps, 5m\n")
    print("| perp | HL | N | 样本起 | 时段 | 均值 | σ | p95\\|x\\| | 半衰期 bars | >20bps | >50bps | HL资金年化% | BN资金年化%(同期) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sym, m in MAP.items():
        hln = m["hl"]
        if not hln:
            continue
        bn = col(load_bn(sym), "c")
        hl = col(load_hl(hln), "c")
        j = pd.concat({"b": bn, "h": hl}, axis=1).dropna()
        spr = cast(pd.Series, np.log(col(j, "b") / col(j, "h"))) * 1e4
        sess = cme_session(pd.DatetimeIndex(j.index))
        hf = hl_funding(hln)
        hf = hf[hf.index >= j.index[0]]
        bf = bn_funding(sym)
        bf = bf[bf.index >= j.index[0]]
        for kind in ("open", "break", "weekend"):
            s = spr[sess == kind]
            if len(s) < 20:
                continue
            print(f"| {sym} | xyz:{hln} | {len(s)} | {cast(pd.Timestamp, j.index[0]).strftime('%m-%d')} | {kind} | {fmt(float(s.mean()))} | {fmt(float(s.std()))} | "
                  f"{fmt(float(s.abs().quantile(0.95)))} | {fmt(half_life(s))} | {fmt(float((s.abs()>20).mean()*100))}% | "
                  f"{fmt(float((s.abs()>50).mean()*100))}% | {fmt(float(hf.mean()*24*365*100),2)} | {fmt(float(bf.mean()*6*365*100),2)} |")


def print_fx() -> None:
    print("\n## FX perps (HL xyz:EUR/GBP/JPY) vs IBKR 现汇 MIDPOINT\n")
    print("| HL | IB | N | 时段 | ln(HL/IB) 均值 bps | σ | p95\\|x\\| | 半衰期 | HL 5m σ bps (open/weekend) | HL资金年化% |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for hln, ibk in FX_HL.items():
        hl = col(load_hl(hln), "c")
        ib = col(load_ib(ibk), "c")
        j = pd.concat({"h": hl, "i": ib}, axis=1).dropna()
        spr = cast(pd.Series, np.log(col(j, "h") / col(j, "i"))) * 1e4
        sess = fx_session(pd.DatetimeIndex(j.index))
        r = cast(pd.Series, np.log(hl)).diff() * 1e4
        rs = fx_session(pd.DatetimeIndex(hl.index))
        hf = hl_funding(hln)
        for kind in ("open",):
            s = spr[sess == kind]
            print(f"| xyz:{hln} | {ibk} | {len(s)} | {kind} | {fmt(float(s.mean()),2)} | {fmt(float(s.std()),2)} | "
                  f"{fmt(float(s.abs().quantile(0.95)),2)} | {fmt(half_life(s))} | "
                  f"{fmt(float(r[rs=='open'].std()),2)}/{fmt(float(r[rs=='weekend'].std()),2)} | {fmt(float(hf.mean()*24*365*100),2)} |")
    print("\n### FX 周末：HL perp 周末移动 vs IB 周日重开缺口\n")
    print("| HL | 周五收 | HL 周末移动 bps | IB 缺口 bps | IB 首60m bps | HL 重开后 60m bps |")
    print("|---|---|---|---|---|---|")
    for hln, ibk in FX_HL.items():
        hl = col(load_hl(hln), "c")
        ib = load_ib(ibk)
        t0 = cast(pd.Timestamp, hl.index[0]).tz_localize(None).normalize()
        t1 = cast(pd.Timestamp, hl.index[-1]).tz_localize(None).normalize()
        for fri in pd.date_range(t0, t1, freq="W-FRI"):
            c = (fri + pd.Timedelta(hours=17)).tz_localize("America/New_York").tz_convert("UTC")
            ro = c + pd.Timedelta(days=2)
            if c < hl.index[0] or ro + pd.Timedelta("2h") > hl.index[-1]:
                continue
            h0 = hl.loc[:c]
            h1 = hl.loc[:ro - pd.Timedelta("5min")]
            h2 = hl.loc[:ro + pd.Timedelta("55min")]
            i0 = ib.loc[:c - pd.Timedelta("5min")]
            i1 = ib.loc[ro:ro + pd.Timedelta("60min")]
            if h0.empty or i0.empty or i1.empty:
                continue
            hm = (h1.iloc[-1] / h0.iloc[-1] - 1) * 1e4
            gap = (i1["o"].iloc[0] / i0["c"].iloc[-1] - 1) * 1e4
            f60 = (i1["c"].iloc[-1] / i1["o"].iloc[0] - 1) * 1e4
            hp = (h2.iloc[-1] / h1.iloc[-1] - 1) * 1e4
            print(f"| xyz:{hln} | {fri.date()} | {hm:.1f} | {gap:.1f} | {f60:.1f} | {hp:.1f} |")


if __name__ == "__main__":
    print_basis()
    print_cross()
    print_fx()
