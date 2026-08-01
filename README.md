# Two-Pair Strategy — SK Hynix KR-line vs US-ADS perp

币安 SKHYNIXUSDT（韩国线）与 SKHYUSDT（美国 ADS 线）永续合约的配对均值回归策略开发目录。

## 策略基线（v3, 2026-08-01 定版）

- 信号变量：汇率调整后对数比价 `lr = ln(KR) − ln(US) − ln(USDKRW)`
- 锚：`lr` 的 24h 滚动均值（288 × 5m bar）
- 波动率：残差按 5 个时段分段（韩股盘中 / 韩→美空窗 / 美股盘中 / 美→韩空窗 / 周末），
  各用同段最近 300 bar 滚动 std —— z=2 在每个时段代表相同的统计罕见度
- 进场 |z| > 2（两腿等美元名义，一次一仓）；离场 |z| < 0.5 或 24h 超时
- funding 现金流按两腿实际结算计入；手续费当前为 0（活动，恢复后需重估）

样本表现（2026-07-10 ~ 08-01，约 3 周）：24 笔，单笔均值 +0.84%，胜率 79%，
累计 +20.05%（按单腿名义），maxDD −5.67%，日度 Sharpe ~7.7（超短样本，仅作版本对比用），
27 格参数网格全部为正。

已验证的设计决策（详见对话/git 历史）：
- 5m 信号采样优于 1m（采样延迟带来的进场过冲是均值回归的隐性优势）
- 汇率调整必须做（7 月 USDKRW 贬值 4%，不调会污染信号）
- 时段感知的硬规则（跳过美股盘中、开盘催化剂离场）在保留全部进场时不稳健，
  已降级为观察项，等更多样本裁决；分段 std 是它们的原则性替代
- 杠杆建议：λ（单腿名义/资金）≤ 2，全仓模式，保证金缓冲 ≥ 3× 历史 MDD

## 文件

- `pair_backtest.py` — 基线回测（含权益曲线 / MDD / Sharpe），funding 实时拉取
- `data/skhx_pair_5m.csv` — 5m 双腿价格 + USDKRW（刷新方式见 git/对话历史中的下载脚本）
- `data/skhx_pair_1m.csv` — 1m 数据（仅研究用，不进信号）
- `data/pair_trades_baseline.csv`、`data/pair_equity_curve.csv` — 回测输出
- `research/` — 前期调研存档（币安/Hyperliquid 股票 perp funding 分析、海力士 funding 专项），
  脚本移动后相对路径未修正，仅作参考

## 下一步

1. 实时监控 / 模拟盘（同一信号函数库，积累样本外记录与时段标签）
2. 平价测试：历史数据喂实盘引擎，断言与回测逐笔一致
3. 自动交易系统设计已定稿（Python + binance-futures-connector + SQLite + Telegram，见对话记录）
