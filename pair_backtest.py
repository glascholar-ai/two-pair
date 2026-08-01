#!/usr/bin/env python3
"""SK Hynix KR-line vs US-ADS perp pair backtest — baseline v3.
5m bars, USDKRW-adjusted log ratio, funding cashflows, and SEGMENT-CONDITIONAL std:
z is normalized by the rolling std of same-session-segment residuals so that a
threshold of 2 means the same statistical rarity in every session regime.
Segments: KR open (00:00-06:30 UTC), KR->US gap, US open (13:30-20:00), US->KR gap, weekend.
Params (WIN=300, Z_IN=2.0, Z_OUT=0.5) sit mid-grid in a 27-cell sensitivity sweep
(all cells positive, sum range +11%..+27% over Jul 10-31 2026)."""
import json, urllib.request, urllib.parse
import pandas as pd, numpy as np

WIN_SEG, Z_IN, Z_OUT, MAX_H, WIN_MU = 300, 2.0, 0.5, 24, 288

df = pd.read_csv("data/skhx_pair_5m.csv", parse_dates=["ts"]).set_index("ts")

def funding(sym):
    rows, cur = [], int(pd.Timestamp("2026-07-08", tz="UTC").timestamp()*1000)
    while True:
        q = urllib.parse.urlencode({"symbol": sym, "startTime": cur, "limit": 1000})
        b = json.load(urllib.request.urlopen(f"https://fapi.binance.com/fapi/v1/fundingRate?{q}"))
        rows += b
        if len(b) < 1000: break
        cur = b[-1]["fundingTime"] + 1
    f = pd.DataFrame(rows)
    f["ts"] = pd.to_datetime(f["fundingTime"], unit="ms", utc=True).dt.round("min")
    f["rate"] = f["fundingRate"].astype(float)
    return f[["ts", "rate"]].drop_duplicates("ts").sort_values("ts")

fk, fu = funding("SKHYNIXUSDT"), funding("SKHYUSDT")

d = df.copy()
d["lr"] = np.log(d.kr) - np.log(d.us) - np.log(d.fx)
d["mu"] = d.lr.rolling(WIN_MU, min_periods=WIN_MU // 2).mean()
d["resid"] = d.lr - d.mu

def seg(ts):
    if ts.weekday() >= 5: return "wknd"
    hm = ts.hour * 60 + ts.minute
    if hm < 390: return "KR_open"
    if hm < 810: return "KR->US"
    if hm < 1200: return "US_open"
    return "US->KR"

d["seg"] = d.index.map(seg)
d["sd_seg"] = d.groupby("seg")["resid"].transform(
    lambda x: x.rolling(WIN_SEG, min_periods=WIN_SEG // 3).std())
d["z"] = d.resid / d.sd_seg
d = d.dropna(subset=["z"])

trades, pos = [], None
for ts, r in d.iterrows():
    if pos is None:
        if abs(r.z) > Z_IN:
            pos = {"entry_ts": ts, "entry_lr": r.lr, "side": -np.sign(r.z),
                   "entry_z": r.z, "regime": r.seg}
    else:
        held = (ts - pos["entry_ts"]).total_seconds() / 3600
        if abs(r.z) < Z_OUT or held >= MAX_H:
            s = pos["side"]
            px = s * (r.lr - pos["entry_lr"]) * 100
            krf = fk[(fk.ts > pos["entry_ts"]) & (fk.ts <= ts)]["rate"].sum()
            usf = fu[(fu.ts > pos["entry_ts"]) & (fu.ts <= ts)]["rate"].sum()
            trades.append({**pos, "exit_ts": ts, "held_h": held, "px_pnl": px,
                           "fund_pnl": (-s * krf + s * usf) * 100,
                           "total": px + (-s * krf + s * usf) * 100,
                           "reason": "conv" if abs(r.z) < Z_OUT else "timeout"})
            pos = None

t = pd.DataFrame(trades)
print(t[["entry_ts", "regime", "entry_z", "held_h", "px_pnl", "fund_pnl", "total", "reason"]]
      .round(3).to_string(index=False))
print("\nn=%d  mean %+.3f%%  median %+.3f%%  win %.0f%%  sum %+.2f%%  worst %+.2f%%" %
      (len(t), t.total.mean(), t.total.median(), (t.total > 0).mean() * 100,
       t.total.sum(), t.total.min()))
print(t.groupby("regime")["total"].agg(["count", "mean", "sum"]).round(3).to_string())
t.round(4).to_csv("data/pair_trades_baseline.csv", index=False)

# ---- bar-level mark-to-market equity curve: max drawdown & Sharpe ----
fk_s, fu_s = fk.set_index("ts")["rate"], fu.set_index("ts")["rate"]
equity, eq, pos2, t0, prev = [], 0.0, None, None, None
for ts, r in d.iterrows():
    if pos2 is not None and prev is not None:
        eq += pos2 * (r.lr - prev[1]) * 100
        for ser, sign in ((fk_s, -1), (fu_s, +1)):
            eq += sign * pos2 * ser[(ser.index > prev[0]) & (ser.index <= ts)].sum() * 100
    if pos2 is None:
        if abs(r.z) > Z_IN:
            pos2, t0 = -np.sign(r.z), ts
    elif abs(r.z) < Z_OUT or (ts - t0).total_seconds() / 3600 >= MAX_H:
        pos2 = None
    equity.append(eq)
    prev = (ts, r.lr)
e = pd.Series(equity, index=d.index)
dd = e - e.cummax()
daily = e.resample("1D").last().dropna().diff().dropna()
n_days = (e.index[-1] - e.index[0]).total_seconds() / 86400
print("\nequity (% of single-leg notional; gross exposure is 2x while in a position):")
print("total %+.2f%%  ann %+.1f%%  Sharpe(daily,365) %.2f  maxDD %.2f%%  Calmar %.1f  in-market %.0f%%" %
      (e.iloc[-1], e.iloc[-1] / n_days * 365, daily.mean() / daily.std() * np.sqrt(365),
       dd.min(), e.iloc[-1] / n_days * 365 / abs(dd.min()), (e.diff().fillna(0) != 0).mean() * 100))
e.to_frame("equity_pct").to_csv("data/pair_equity_curve.csv")
