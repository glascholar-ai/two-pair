#!/usr/bin/env python3
"""Screen Binance equity perps for pair-trading candidates: 4h log-return
correlation + spread half-life on top pairs."""
import json, time, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data" / "pairs"
OUT.mkdir(parents=True, exist_ok=True)

d = json.load(urllib.request.urlopen("https://fapi.binance.com/fapi/v1/exchangeInfo"))
SYMS = sorted(s["symbol"] for s in d["symbols"]
              if s.get("contractType") == "TRADIFI_PERPETUAL" and s.get("status") == "TRADING"
              and s.get("underlyingType") in ("EQUITY", "PREMARKET", "KR_EQUITY", "HK_EQUITY", "JP_EQUITY"))

closes = {}
for i, s in enumerate(SYMS):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={s}&interval=4h&limit=500"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            k = json.load(r)
        if len(k) >= 120:  # ≥20 days of history
            ser = pd.Series({int(x[0]): float(x[4]) for x in k})
            closes[s[:-4]] = ser
    except Exception as ex:
        print(f"[{s}] {ex!r}", flush=True)
    time.sleep(0.08)
print(f"fetched {len(closes)}/{len(SYMS)} with enough history", flush=True)

px = pd.DataFrame(closes)
px.index = pd.to_datetime(px.index, unit="ms", utc=True)
px.to_csv(OUT / "px_4h.csv")
rets = np.log(px).diff()

pairs = []
cols = list(px.columns)
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        a, b = cols[i], cols[j]
        r = rets[[a, b]].dropna()
        if len(r) < 180:
            continue
        c = r[a].corr(r[b])
        if c < 0.55:
            continue
        # spread half-life via AR(1) on log price ratio
        lp = (np.log(px[a]) - np.log(px[b])).dropna()
        lp = lp.loc[r.index]
        dz = lp.diff().dropna()
        z1 = lp.shift(1).dropna().loc[dz.index]
        z1c = z1 - z1.mean()
        phi = (z1c * dz).sum() / (z1c ** 2).sum()
        hl_bars = -np.log(2) / phi if phi < 0 else np.inf
        spread_std_bps = (lp - lp.rolling(180).mean()).std() * 1e4
        pairs.append({"a": a, "b": b, "corr": c, "n": len(r),
                      "halflife_h": hl_bars * 4, "spread_std_bps": spread_std_bps})

pf = pd.DataFrame(pairs).sort_values("corr", ascending=False)
pf.to_csv(OUT / "pair_screen.csv", index=False)
pd.set_option("display.width", 200)
print(pf.head(60).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
print(f"\ntotal pairs corr>0.55: {len(pf)}")
