#!/usr/bin/env python3
"""Analyze Binance stock-perp funding: top-5 by daily average, session vs off-session."""
import argparse
import pandas as pd
import datetime as dt

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=None, help="restrict to last N days (default: all data)")
ap.add_argument("--top", type=int, default=5, help="rank top N symbols")
ap.add_argument("--min-days", type=int, default=14, help="min distinct days of data to be ranked")
args = ap.parse_args()

df = pd.read_csv("data/funding_history.csv")
df["fundingRate"] = df["fundingRate"].astype(float)
df["ts"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.round("min")
df["date"] = df["ts"].dt.date
df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)

# accrual window from actual consecutive settlements — compute BEFORE any window
# filter so the first event inside the window keeps its true previous settlement
df["prev_ts"] = df.groupby("symbol")["ts"].shift(1)
df["interval_h"] = ((df["ts"] - df["prev_ts"]).dt.total_seconds() / 3600).clip(upper=8).fillna(8)
df["win_start"] = df["ts"] - pd.to_timedelta(df["interval_h"], unit="h")

if args.days:
    cutoff = df["ts"].max() - pd.Timedelta(days=args.days)
    df = df[df["ts"] > cutoff]
    print(f"window: {cutoff} -> {df['ts'].max()}  ({args.days} days)")

MIN_DAYS = args.min_days
TOP_N = args.top

# ---- daily average funding per symbol ----
daily = df.groupby(["symbol", "underlyingType", "date"])["fundingRate"].sum().reset_index()
stats = daily.groupby(["symbol", "underlyingType"]).agg(
    days=("date", "nunique"),
    daily_avg=("fundingRate", "mean"),
).reset_index()
stats["daily_avg_pct"] = stats["daily_avg"] * 100
stats["annualized_pct"] = stats["daily_avg"] * 365 * 100

eligible = stats[stats["days"] >= MIN_DAYS].sort_values("daily_avg", ascending=False)
print("=== Top %d by daily avg funding (>=%d days of data) ===" % (TOP_N, MIN_DAYS))
print(eligible.head(TOP_N)[["symbol", "underlyingType", "days", "daily_avg_pct", "annualized_pct"]].round(4).to_string(index=False))
print()
print("=== Bottom 5 (most negative) ===")
print(eligible.tail(5)[["symbol", "underlyingType", "days", "daily_avg_pct", "annualized_pct"]].round(4).to_string(index=False))
print()
excl = stats[stats["days"] < MIN_DAYS].sort_values("daily_avg", ascending=False)
if len(excl):
    print("=== Excluded (<%d days of data), top few ===" % MIN_DAYS)
    print(excl.head(6)[["symbol", "underlyingType", "days", "daily_avg_pct"]].round(4).to_string(index=False))

top5 = eligible.head(TOP_N)["symbol"].tolist()
print("\nTop %d:" % TOP_N, top5)

# ---- session classification by accrual-window overlap ----
# Each funding payment settles at ts and accrues over (prev_settlement, ts].
# Overlap that window with the underlying cash-market session:
#   US (EDT, May-Jul 2026): 13:30-20:00 UTC on US trading days
#   KR: 09:00-15:30 KST = 00:00-06:30 UTC on KR trading days
US_HOLIDAYS = {dt.date(2026, 5, 25), dt.date(2026, 6, 19), dt.date(2026, 7, 3)}
KR_HOLIDAYS = {dt.date(2026, 5, 25), dt.date(2026, 6, 3)}
SESSIONS = {
    "EQUITY": (dt.time(13, 30), dt.time(20, 0), US_HOLIDAYS),
    "KR_EQUITY": (dt.time(0, 0), dt.time(6, 30), KR_HOLIDAYS),
    "HK_EQUITY": (dt.time(1, 30), dt.time(8, 0), None),  # 09:30-16:00 HKT approx, ignoring lunch break
}

def overlap_hours(win_start, win_end, utype):
    t0, t1, hols = SESSIONS[utype]
    total = pd.Timedelta(0)
    d = win_start.date()
    while d <= win_end.date():
        if d.weekday() < 5 and (hols is None or d not in hols):
            s = pd.Timestamp.combine(d, t0).tz_localize("UTC")
            e = pd.Timestamp.combine(d, t1).tz_localize("UTC")
            lo, hi = max(s, win_start), min(e, win_end)
            if hi > lo:
                total += hi - lo
        d += dt.timedelta(days=1)
    return total.total_seconds() / 3600

t5 = df[df["symbol"].isin(top5)].copy()
t5["overlap_h"] = t5.apply(lambda r: overlap_hours(r["win_start"], r["ts"], r["underlyingType"]), axis=1)
t5["seg"] = (t5["overlap_h"] > 0).map({True: "session", False: "off"})

print("\n=== Session-overlapping vs off-session funding windows, top %d ===" % TOP_N + "")
print("(rate figures in %, per settlement event; sum = total funding paid over the period)")
seg = t5.groupby(["symbol", "seg"]).agg(
    events=("fundingRate", "size"),
    hours=("interval_h", "sum"),
    mean_pct=("fundingRate", lambda x: x.mean() * 100),
    sum_pct=("fundingRate", lambda x: x.sum() * 100),
    max_pct=("fundingRate", lambda x: x.max() * 100),
    min_pct=("fundingRate", lambda x: x.min() * 100),
)
seg["pct_per_hour"] = seg["sum_pct"] / seg["hours"]
print(seg.round(4).to_string())

print("\n=== Mean funding by settlement hour (UTC), top %d, %% ===" % TOP_N + "")
byhr = t5.pivot_table(index="symbol", columns=t5["ts"].dt.hour, values="fundingRate", aggfunc="mean") * 100
print(byhr.round(4).to_string())

print("\n=== Weekday vs weekend (by window start), top %d, mean %% per event ===" % TOP_N + "")
t5["wknd"] = t5["win_start"].dt.weekday >= 5
wk = t5.pivot_table(index="symbol", columns="wknd", values="fundingRate", aggfunc="mean") * 100
wk.columns = ["weekday", "weekend"]
print(wk.round(4).to_string())

stats.sort_values("daily_avg", ascending=False).to_csv("data/summary_all_symbols.csv", index=False)
t5.to_csv("data/top5_detail.csv", index=False)
print("\nsaved data/summary_all_symbols.csv, data/top5_detail.csv")
