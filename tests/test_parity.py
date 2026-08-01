"""Parity tests: the shared engine must reproduce the pandas reference.

Two layers:
  1. An independent pandas implementation of the baseline (a faithful port
     of the pre-library research script) is run over the real dataset and
     must produce the SAME trade list as twopair.backtest.
  2. Known headline numbers of baseline v3 are pinned so accidental
     parameter drift fails loudly.

These tests need data/skhx_pair_5m.csv and the funding CSVs; they skip if
absent (e.g. a fresh clone without data).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from twopair import data as datamod
from twopair.backtest import run_backtest
from twopair.config import Config

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
PAIR_CSV = DATA / "skhx_pair_5m.csv"
FUND_KR = DATA / "funding_kr.csv"
FUND_US = DATA / "funding_us.csv"

pytestmark = pytest.mark.skipif(
    not (PAIR_CSV.exists() and FUND_KR.exists() and FUND_US.exists()),
    reason="historical dataset not present")


def _reference_trades(pair: pd.DataFrame, fk: pd.Series, fu: pd.Series,
                      cfg: Config) -> pd.DataFrame:
    """Independent pandas implementation of baseline v3 (research port)."""
    d = pair.copy()
    d["lr"] = np.log(d.kr) - np.log(d.us) - np.log(d.fx)
    d["mu"] = d.lr.rolling(cfg.win_mu, min_periods=cfg.min_mu).mean()
    d["resid"] = d.lr - d.mu

    def seg(ts: pd.Timestamp) -> str:
        if ts.weekday() >= 5:
            return "wknd"
        hm = ts.hour * 60 + ts.minute
        if hm < 390:
            return "KR_open"
        if hm < 810:
            return "KR->US"
        if hm < 1200:
            return "US_open"
        return "US->KR"

    d["seg"] = d.index.map(seg)
    d["sd"] = d.groupby("seg")["resid"].transform(
        lambda x: x.rolling(cfg.win_sd, min_periods=cfg.min_sd).std())
    d["z"] = d.resid / d.sd

    trades = []
    pos = None
    need_rearm = False
    prev = None
    for ts, r in d.iterrows():
        if pos is not None and prev is not None:
            step = pos["s"] * (r.lr - prev[1]) * 100
            krf = fk[(fk.index > prev[0]) & (fk.index <= ts)].sum()
            usf = fu[(fu.index > prev[0]) & (fu.index <= ts)].sum()
            step += pos["s"] * (-krf + usf) * 100
            pos["mtm"] += step
        if not np.isnan(r.z):
            if need_rearm and abs(r.z) < cfg.z_in:
                need_rearm = False
            if pos is None:
                if not need_rearm and abs(r.z) > cfg.z_in:
                    pos = {"ts0": ts, "s": -np.sign(r.z), "mtm": 0.0}
            else:
                held = (ts - pos["ts0"]).total_seconds() / 3600
                stop = (cfg.mtm_stop_pct > 0
                        and pos["mtm"] <= -cfg.mtm_stop_pct)
                if abs(r.z) < cfg.z_out or held >= cfg.max_hold_hours or stop:
                    reason = ("conv" if abs(r.z) < cfg.z_out
                              else ("stop" if stop else "timeout"))
                    if stop:
                        need_rearm = True
                    trades.append({"entry_ts": pos["ts0"], "exit_ts": ts,
                                   "side": int(pos["s"]),
                                   "pnl_pct": pos["mtm"], "reason": reason})
                    pos = None
        prev = (ts, r.lr)
    return pd.DataFrame(trades)


@pytest.fixture(scope="module")
def dataset() -> tuple:
    pair = datamod.load_pair_csv(str(PAIR_CSV))
    fk = datamod.load_funding_csv(str(FUND_KR))
    fu = datamod.load_funding_csv(str(FUND_US))
    return pair, fk, fu


class TestEngineMatchesPandasReference:
    @pytest.mark.parametrize("stop", [2.5, 0.0])
    def test_trade_lists_identical(self, dataset: tuple, stop: float) -> None:
        pair, fk, fu = dataset
        cfg = Config(mtm_stop_pct=stop)
        ref = _reference_trades(pair, fk, fu, cfg)
        result = run_backtest(pair, fk, fu, cfg)
        got = result.trades_frame()
        assert len(got) == len(ref)
        for i in range(len(ref)):
            assert got.entry_ts.iloc[i] == ref.entry_ts.iloc[i]
            assert got.exit_ts.iloc[i] == ref.exit_ts.iloc[i]
            assert got.side.iloc[i] == ref.side.iloc[i]
            assert got.reason.iloc[i] == ref.reason.iloc[i]
            assert got.pnl_pct.iloc[i] == pytest.approx(
                ref.pnl_pct.iloc[i], abs=1e-9)


class TestBaselinePinnedNumbers:
    """Headline numbers of baseline v3 on the frozen Jul 10 - Aug 1 dataset."""

    def test_stop_25(self, dataset: tuple) -> None:
        pair, fk, fu = dataset
        result = run_backtest(pair, fk, fu, Config(mtm_stop_pct=2.5))
        frame = result.trades_frame()
        assert len(frame) == 26
        assert frame.pnl_pct.sum() == pytest.approx(16.51, abs=0.05)
        assert (frame.reason == "stop").sum() == 4
        assert result.max_drawdown_pct == pytest.approx(-4.18, abs=0.05)

    def test_stop_off(self, dataset: tuple) -> None:
        pair, fk, fu = dataset
        result = run_backtest(pair, fk, fu, Config(mtm_stop_pct=0.0))
        frame = result.trades_frame()
        assert len(frame) == 24
        assert frame.pnl_pct.sum() == pytest.approx(20.05, abs=0.05)
        assert frame.pnl_pct.min() == pytest.approx(-1.91, abs=0.02)
        assert result.max_drawdown_pct == pytest.approx(-5.67, abs=0.05)
