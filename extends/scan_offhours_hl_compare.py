#!/usr/bin/env python3
"""Binance vs Hyperliquid (xyz HIP-3) dead-zone spike reversal, same names, same window.

HL candleSnapshot only serves the most recent ~5000 5m bars (~17 days), so the
comparison is restricted to that overlap window on both venues.
Run: python scan_offhours_hl_compare.py
"""
from __future__ import annotations

from typing import Dict, List, cast

import pandas as pd

from scan_offhours_reversal import (ASIA_LINKED, D_HL, NON_US, load_ohlc, sim_limits,
                                    spike_events, summarize, trade_stats)


def hl_names() -> List[str]:
    return sorted(p.stem[4:-3] for p in D_HL.glob("xyz_*_5m.parquet")
                  if p.stem[4:-3] not in NON_US and p.stem[4:-3] not in ASIA_LINKED)


def main() -> None:
    pd.set_option("display.width", 200)
    names = hl_names()
    hl: Dict[str, pd.DataFrame] = {}
    bn: Dict[str, pd.DataFrame] = {}
    for n in names:
        h, b = load_ohlc("hl", n), load_ohlc("bn", n)
        if h is None or b is None or len(h) < 1000:
            continue
        t0, t1 = h.index[0], min(h.index[-1], b.index[-1])
        # keep 3 extra days of BN history before t0 so the rolling vol is warm on both
        hl[n] = cast(pd.DataFrame, h[(h.index >= t0) & (h.index <= t1)])
        bn[n] = cast(pd.DataFrame, b[(b.index >= t0 - pd.Timedelta(days=3)) & (b.index <= t1)])
    print(f"names={len(hl)} window {min(d.index[0] for d in hl.values()):%Y-%m-%d} .. "
          f"{max(d.index[-1] for d in hl.values()):%Y-%m-%d}")
    print("names:", " ".join(sorted(hl)))
    for lab, data in (("HL", hl), ("BN", bn)):
        ev = pd.concat([spike_events(df, s) for s, df in data.items()])
        e4 = cast(pd.DataFrame, ev[cast(pd.Series, ev["z"]).abs() > 4])
        print(f"\n== {lab}: |z|>4 close-to-close spikes by segment ==")
        print(summarize(e4, "seg").to_string())
        w = cast(pd.DataFrame, ev[ev["zw"] > 4])
        print(f"== {lab}: wick spikes zw>4 by segment ==")
        print(summarize(w, "seg", "wdir").to_string())
        # 5m vol level in DEAD (median sigma bps) for context
        print(f"== {lab}: median 3d 5m sigma (bps) at DEAD events: "
              f"{ev.loc[ev['seg'] == 'DEAD', 'vol'].median() * 1e4:.1f}")
        rows = []
        for k in (3.0, 4.0):
            trs = [sim_limits(df, s, k, 36, segs=("DEAD", "WKND", "AH", "PRE", "OPEN", "REST"))
                   for s, df in data.items()]
            tr = pd.concat([t for t in trs if len(t)])
            for sg, e in tr.groupby("seg"):
                st = trade_stats(e)
                st["k"], st["seg"] = k, sg
                rows.append(st)
        print(f"== {lab}: limit sim (hold 36) ==")
        print(pd.DataFrame(rows).set_index(["k", "seg"])[
            ["fills", "fills/day", "gross_bps", "net_bps", "win%", "p5_net", "min_net"]].round(1).to_string())


if __name__ == "__main__":
    main()
