#!/usr/bin/env python3
"""SK Hynix KR-line vs US-ADS perp pair backtest — baseline v3.
5m bars, USDKRW-adjusted log ratio, funding cashflows, segment-conditional std,
and an optional mark-to-market stop.

Signal: lr = ln(KR) - ln(US) - ln(USDKRW); anchor = 24h rolling mean (288 bars,
one full session cycle — do not change off 24h multiples, the ratio has intraday
seasonality); sd = rolling std of anchor residuals over the last 300 bars of the
SAME session segment (KR open / KR->US gap / US open / US->KR gap / weekend).
Enter |z| > 2 (equal-notional legs, one position at a time); exit |z| < 0.5,
24h timeout, or MTM stop.

Params sit on the ridge of a 27-cell sweep (all cells positive) plus separate
mu/sd window sweeps (mu peaked at 24h; sd safe zone 300-450)."""
import argparse
import json, urllib.request, urllib.parse
import pandas as pd, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mtm-stop", type=float, default=2.5,
                help="close when trade MTM (price+funding, %% of single-leg notional) "
                     "falls below -X; 0 disables (default 2.5)")
args = ap.parse_args()

WIN_SEG, Z_IN, Z_OUT, MAX_H, WIN_MU = 300, 2.0, 0.5, 24, 288
MTM_STOP = args.mtm_stop

df = pd.read_csv("data/skhx_pair_5m.csv", parse_dates=["ts"]).set_index("ts")

def funding(sym):
    rows, cur = [], int(pd.Timestamp("2026-07-08", tz="UTC").timestamp() * 1000)
    while True:
        q = urllib.parse.urlencode({"symbol": sym, "startTime": cur, "limit": 1000})
        b = json.load(urllib.request.urlopen(f"https://fapi.binance.com/fapi/v1/fundingRate?{q}"))
        rows += b
        if len(b) < 1000: break
        cur = b[-1]["fundingTime"] + 1
    f = pd.DataFrame(rows)
    f["ts"] = pd.to_datetime(f["fundingTime"], unit="ms", utc=True).dt.round("min")
    f["rate"] = f["fundingRate"].astype(float)
    return f[["ts", "rate"]].drop_duplicates("ts").set_index("ts")["rate"].sort_index()

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

# single pass produces both the trade list and the bar-level equity curve
trades, pos = [], None
equity, eq, prev = [], 0.0, None
need_reset = False   # after a stop, wait for |z| to re-enter the band before re-arming
for ts, r in d.iterrows():
    if pos is not None and prev is not None:
        step = pos["s"] * (r.lr - prev[1]) * 100
        for ser, sign in ((fk, -1), (fu, +1)):
            step += sign * pos["s"] * ser[(ser.index > prev[0]) & (ser.index <= ts)].sum() * 100
        pos["mtm"] += step
        eq += step
    if need_reset and abs(r.z) < Z_IN:
        need_reset = False
    if pos is None:
        if not need_reset and abs(r.z) > Z_IN:
            pos = {"entry_ts": ts, "entry_lr": r.lr, "s": -np.sign(r.z),
                   "entry_z": r.z, "regime": r.seg, "mtm": 0.0, "maxz": abs(r.z)}
    else:
        pos["maxz"] = max(pos["maxz"], abs(r.z))
        held = (ts - pos["entry_ts"]).total_seconds() / 3600
        stop_hit = MTM_STOP > 0 and pos["mtm"] <= -MTM_STOP
        if abs(r.z) < Z_OUT or held >= MAX_H or stop_hit:
            reason = "conv" if abs(r.z) < Z_OUT else ("stop" if stop_hit else "timeout")
            if stop_hit:
                need_reset = True
            trades.append({"entry_ts": pos["entry_ts"], "regime": pos["regime"],
                           "entry_z": pos["entry_z"], "maxz": pos["maxz"], "exit_ts": ts,
                           "held_h": held, "total": pos["mtm"], "reason": reason})
            pos = None
    equity.append(eq)
    prev = (ts, r.lr)

t = pd.DataFrame(trades)
print(t[["entry_ts", "regime", "entry_z", "maxz", "held_h", "total", "reason"]]
      .round(3).to_string(index=False))
print("\nn=%d  mean %+.3f%%  median %+.3f%%  win %.0f%%  sum %+.2f%%  worst %+.2f%%  stops %d" %
      (len(t), t.total.mean(), t.total.median(), (t.total > 0).mean() * 100,
       t.total.sum(), t.total.min(), (t.reason == "stop").sum()))
print(t.groupby("regime")["total"].agg(["count", "mean", "sum"]).round(3).to_string())

e = pd.Series(equity, index=d.index)
dd = e - e.cummax()
daily = e.resample("1D").last().dropna().diff().dropna()
n_days = (e.index[-1] - e.index[0]).total_seconds() / 86400
print("\nequity (%% of single-leg notional; gross exposure is 2x while in a position; mtm_stop=%s):" %
      (MTM_STOP if MTM_STOP > 0 else "off"))
print("total %+.2f%%  ann %+.1f%%  Sharpe(daily,365) %.2f  maxDD %.2f%%  Calmar %.1f  in-market %.0f%%" %
      (e.iloc[-1], e.iloc[-1] / n_days * 365, daily.mean() / daily.std() * np.sqrt(365),
       dd.min(), e.iloc[-1] / n_days * 365 / abs(dd.min()), (e.diff().fillna(0) != 0).mean() * 100))
t.round(4).to_csv("data/pair_trades_baseline.csv", index=False)
e.to_frame("equity_pct").to_csv("data/pair_equity_curve.csv")
