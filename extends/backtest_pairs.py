#!/usr/bin/env python3
"""Z-score pair-trading backtest on Binance equity-perp 5m data, net of fees + funding.

Signal on bar close, execution at next bar open. Dollar-neutral 1:1 log-ratio pairs.
"""
import numpy as np
import pandas as pd
from pathlib import Path

D = Path(__file__).parent / "data" / "bn5m"
OUT = Path(__file__).parent / "data" / "pairs"

LEV = {"MUU", "MVLL", "INTW", "SNXX", "TQQQ", "SQQQ", "SOXL", "SOXS", "TZA", "TMF",
       "UVXY", "KORU", "CSOPSAMSUNG2L", "CSOPSKHYNIX2L", "BITO", "BSP", "MSTX"}
DUP = {("SKHY", "SKHYNIX"), ("SPCX", "SPCXUSD1")}

PAIRS = [  # curated from the 4h screen (tier 1+2)
    ("AMAT", "LRCX"), ("KLAC", "LRCX"), ("AMAT", "KLAC"),
    ("TSM", "EWT"), ("EWY", "SAMSUNG"),
    ("MU", "DRAM"), ("DRAM", "SKHYNIX"), ("MU", "SKHYNIX"), ("SAMSUNG", "SKHYNIX"),
    ("COHR", "LITE"), ("AAOI", "LITE"), ("AAOI", "COHR"),
    ("WDC", "STXX"), ("WDC", "SNDK"), ("MU", "SNDK"),
    ("CRWV", "NBIS"), ("ASTS", "RKLB"),
    ("HOOD", "COIN"), ("MSTR", "COIN"),
    ("ALAB", "CRDO"), ("AMD", "NVDA"), ("QQQ", "SPY"),
]

COST_RT = {"taker": 20.0, "maker": 4.0}   # bps, 4 legs incl. slippage
Z_EXIT, Z_STOP = 0.5, 4.0
MAX_HOLD_BARS = 288 * 14                   # 14 days


def load_sym(sym):
    f = D / f"{sym}USDT.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("dt")[["o", "c"]]


def load_funding():
    f = pd.read_parquet(D / "_funding.parquet")
    f["fundingRate"] = f["fundingRate"].astype(float)
    f["dt"] = pd.to_datetime(f["fundingTime"].astype("int64"), unit="ms", utc=True)
    f["sym"] = f["symbol"].str[:-4]
    return {s: g.set_index("dt")["fundingRate"].sort_index() for s, g in f.groupby("sym")}


def run_pair(a, b, pa, pb, fund, window, z_in, cost_rt):
    df = pa.join(pb, lsuffix="_a", rsuffix="_b", how="inner").dropna()
    if len(df) < window + 288 * 10:
        return None
    lr = np.log(df["c_a"] / df["c_b"])
    mu = lr.rolling(window).mean()
    sd = lr.rolling(window).std()
    z = ((lr - mu) / sd).to_numpy()
    o_a, o_b = df["o_a"].to_numpy(), df["o_b"].to_numpy()
    idx = df.index
    fa = fund.get(a, pd.Series(dtype=float))
    fb = fund.get(b, pd.Series(dtype=float))

    trades, pos, n = [], 0, len(df)
    for i in range(window, n - 1):
        if pos == 0:
            if z[i] > z_in:
                pos, i0 = -1, i + 1   # short A / long B
            elif z[i] < -z_in:
                pos, i0 = 1, i + 1    # long A / short B
        else:
            stop = abs(z[i]) > Z_STOP and np.sign(z[i]) == -pos
            if abs(z[i]) < Z_EXIT or stop or (i - i0) >= MAX_HOLD_BARS or i == n - 2:
                i1 = i + 1
                gross = pos * ((np.log(o_a[i1] / o_a[i0]) - np.log(o_b[i1] / o_b[i0])) * 1e4)
                fw_a = fa.loc[idx[i0]:idx[i1]].sum() if len(fa) else 0.0
                fw_b = fb.loc[idx[i0]:idx[i1]].sum() if len(fb) else 0.0
                fcost = (pos * fw_a - pos * fw_b) * 1e4   # long leg pays +f
                trades.append({"pair": f"{a}/{b}", "t_in": idx[i0], "t_out": idx[i1],
                               "dir": pos, "hold_d": (i1 - i0) / 288,
                               "gross": gross, "funding": -fcost,
                               "net": gross - fcost - cost_rt, "stopped": bool(stop)})
                pos = 0
    return pd.DataFrame(trades)


def main():
    fund = load_funding()
    px = {}
    for s in sorted({x for p in PAIRS for x in p}):
        v = load_sym(s)
        if v is not None:
            px[s] = v

    all_rows, best_trades = [], []
    for a, b in PAIRS:
        if a not in px or b not in px:
            print(f"skip {a}/{b} (missing data)")
            continue
        best = None
        for window in (288 * 5, 288 * 10):
            for z_in in (2.0, 2.5):
                for fee_name, cost in COST_RT.items():
                    tr = run_pair(a, b, px[a], px[b], fund, window, z_in, cost)
                    if tr is None or not len(tr):
                        continue
                    row = {"pair": f"{a}/{b}", "window_d": window // 288, "z_in": z_in,
                           "fees": fee_name, "trades": len(tr),
                           "win%": (tr.net > 0).mean() * 100,
                           "net_med": tr.net.median(), "net_total": tr.net.sum(),
                           "fund_total": tr.funding.sum(),
                           "worst": tr.net.min(), "hold_med_d": tr.hold_d.median(),
                           "sharpe_tr": tr.net.mean() / tr.net.std() if tr.net.std() > 0 else np.nan}
                    all_rows.append(row)
                    if fee_name == "taker" and (best is None or row["net_total"] > best[0]["net_total"]):
                        best = (row, tr)
        if best:
            best_trades.append(best[1].assign(window_d=best[0]["window_d"], z_in=best[0]["z_in"]))

    res = pd.DataFrame(all_rows)
    res.to_csv(OUT / "pair_bt_results.csv", index=False)
    bt = pd.concat(best_trades, ignore_index=True)
    bt.to_csv(OUT / "pair_bt_trades.csv", index=False)

    pd.set_option("display.width", 250)
    fmt = lambda x: f"{x:,.1f}"
    print("=== best taker-config per pair (by net_total) ===")
    tk = res[res.fees == "taker"].sort_values("net_total", ascending=False)
    print(tk.loc[tk.groupby("pair")["net_total"].idxmax()]
          .sort_values("net_total", ascending=False).to_string(index=False, float_format=fmt))
    print("\n=== same configs, maker fees ===")
    mk = res[res.fees == "maker"]
    print(mk.loc[mk.groupby("pair")["net_total"].idxmax()]
          .sort_values("net_total", ascending=False).to_string(index=False, float_format=fmt))
    print("\n=== portfolio (best taker configs), monthly net bps ===")
    bt["month"] = pd.to_datetime(bt.t_out, utc=True).dt.to_period("M").astype(str)
    print(bt.groupby("month")["net"].agg(["count", "sum"]).to_string(float_format=fmt))


if __name__ == "__main__":
    main()
