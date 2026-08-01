#!/usr/bin/env python3
"""Analyze Hyperliquid stock-perp funding: daily averages, session vs off-session,
and cross-exchange comparison with Binance."""
import argparse
import pandas as pd
import datetime as dt

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=None)
ap.add_argument("--top", type=int, default=10)
ap.add_argument("--min-days", type=int, default=14)
args = ap.parse_args()

df = pd.read_csv("data/hl_funding_history.csv")
df["fundingRate"] = df["fundingRate"].astype(float)
df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.round("min")
df = df[~df["coin"].str.split(":").str[1].isin({"TOTAL2","OTHERS","BTCD"})]
df = df.sort_values(["coin", "ts"]).reset_index(drop=True)
if args.days:
    cutoff = df["ts"].max() - pd.Timedelta(days=args.days)
    df = df[df["ts"] > cutoff]
    print(f"window: {cutoff} -> {df['ts'].max()}")
df["date"] = df["ts"].dt.date

# ---- daily average funding ----
daily = df.groupby(["coin", "market", "date"])["fundingRate"].sum().reset_index()
stats = daily.groupby(["coin", "market"]).agg(days=("date", "nunique"), daily_avg=("fundingRate", "mean")).reset_index()
stats["daily_avg_pct"] = stats["daily_avg"] * 100
stats["annualized_pct"] = stats["daily_avg"] * 365 * 100

eligible = stats[stats["days"] >= args.min_days].sort_values("daily_avg", ascending=False)
print(f"=== HL top {args.top} by daily avg funding (>= {args.min_days} days) ===")
print(eligible.head(args.top)[["coin", "market", "days", "daily_avg_pct", "annualized_pct"]].round(4).to_string(index=False))
print("\n=== HL bottom 5 (most negative) ===")
print(eligible.tail(5)[["coin", "market", "days", "daily_avg_pct", "annualized_pct"]].round(4).to_string(index=False))

# ---- session classification (hourly windows: (ts-1h, ts]) ----
HOLIDAYS = {
    "US": {dt.date(2026, 5, 25), dt.date(2026, 6, 19), dt.date(2026, 7, 3)},
    "KR": {dt.date(2026, 5, 25), dt.date(2026, 6, 3)},
    "JP": {dt.date(2026, 7, 20)},                       # Marine Day
    "HK": {dt.date(2026, 6, 19), dt.date(2026, 7, 1)},  # Dragon Boat, HKSAR Day
    "CN": {dt.date(2026, 6, 19)},                       # Dragon Boat
}
SESSIONS = {
    "US": (dt.time(13, 30), dt.time(20, 0)),
    "KR": (dt.time(0, 0), dt.time(6, 30)),
    "JP": (dt.time(0, 0), dt.time(6, 0)),
    "HK": (dt.time(1, 30), dt.time(8, 0)),
    "CN": (dt.time(1, 30), dt.time(7, 0)),
}

def in_session(ts, mkt):
    t0, t1 = SESSIONS[mkt]
    ws, we = ts - pd.Timedelta(hours=1), ts
    d = ws.date()
    if d.weekday() >= 5 or d in HOLIDAYS[mkt]:
        in_prev = False
    else:
        s = pd.Timestamp.combine(d, t0).tz_localize("UTC")
        e = pd.Timestamp.combine(d, t1).tz_localize("UTC")
        in_prev = max(s, ws) < min(e, we)
    if in_prev:
        return True
    d2 = we.date()
    if d2 != d and d2.weekday() < 5 and d2 not in HOLIDAYS[mkt]:
        s = pd.Timestamp.combine(d2, t0).tz_localize("UTC")
        e = pd.Timestamp.combine(d2, t1).tz_localize("UTC")
        return max(s, ws) < min(e, we)
    return False

topN = eligible.head(args.top)["coin"].tolist()
t = df[df["coin"].isin(topN)].copy()
t["seg"] = t.apply(lambda r: "session" if in_session(r["ts"], r["market"]) else "off", axis=1)

print(f"\n=== Session vs off-session, HL top {args.top} (%/hour) ===")
seg = t.groupby(["coin", "seg"])["fundingRate"].agg(["count", "mean", "sum"])
seg[["mean", "sum"]] *= 100
seg = seg.rename(columns={"mean": "mean_pct_per_h", "sum": "sum_pct"})
print(seg.round(4).to_string())

t["wknd"] = (t["ts"] - pd.Timedelta(hours=1)).dt.weekday >= 5
wk = t.pivot_table(index="coin", columns="wknd", values="fundingRate", aggfunc="mean") * 100
wk.columns = ["weekday_pct_h", "weekend_pct_h"]
print(f"\n=== Weekday vs weekend, HL top {args.top} (mean %/hour) ===")
print(wk.round(4).to_string())

# ---- cross-exchange comparison with Binance ----
try:
    bn = pd.read_csv("data/funding_history.csv")
except FileNotFoundError:
    bn = None
if bn is not None:
    bn["fundingRate"] = bn["fundingRate"].astype(float)
    bn["ts"] = pd.to_datetime(bn["fundingTime"], unit="ms", utc=True)
    if args.days:
        bn = bn[bn["ts"] > bn["ts"].max() - pd.Timedelta(days=args.days)]
    bn["date"] = bn["ts"].dt.date
    bnd = bn.groupby(["symbol", "date"])["fundingRate"].sum().reset_index()
    bstats = bnd.groupby("symbol").agg(bn_days=("date", "nunique"), bn_daily=("fundingRate", "mean")).reset_index()

    SPECIAL = {"SKHYNIX": "SKHX", "SAMSUNG": "SMSN"}
    bstats["base"] = bstats["symbol"].str.replace(r"(USDT|USD1)$", "", regex=True).map(lambda b: SPECIAL.get(b, b))
    hl = stats.copy()
    hl["base"] = hl["coin"].str.split(":").str[1]
    hl = hl[hl["coin"].str.startswith("xyz:")]  # avoid para/xyz dupes
    m = bstats.merge(hl, on="base", how="inner")
    m = m[(m["bn_days"] >= min(args.min_days, 7)) & (m["days"] >= min(args.min_days, 7))]
    m["bn_daily_pct"] = m["bn_daily"] * 100
    m["spread_pct"] = m["bn_daily_pct"] - m["daily_avg_pct"]
    m = m.sort_values("spread_pct", ascending=False)
    print("\n=== Binance vs Hyperliquid daily avg funding, overlapping names (%) ===")
    print(m[["base", "market", "bn_daily_pct", "daily_avg_pct", "spread_pct"]]
          .rename(columns={"daily_avg_pct": "hl_daily_pct"}).round(4).to_string(index=False))

stats.sort_values("daily_avg", ascending=False).to_csv("data/hl_summary.csv", index=False)
t.to_csv("data/hl_top_detail.csv", index=False)
print("\nsaved data/hl_summary.csv, data/hl_top_detail.csv")
