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

## 代码结构（v1.0）

```
twopair/               策略库 —— 回测与实盘共用同一信号路径
  config.py            全部参数（frozen dataclass + JSON/env 加载）
  signal.py            SignalEngine：增量 rolling 统计、时段分段、z（纯函数，无 I/O）
  strategy.py          Strategy 状态机：进出场、MTM 止损、再武装、告警
  data.py              币安 klines/funding、Yahoo FX、数据集组装
  backtest.py          回测 runner（复用 SignalEngine + Strategy）
  live.py              LiveApp：5 分钟轮询循环（warmup → 每轮交易所对账 → step）
  executor.py          LiveExecutor（BBO 被动追单、断腿修复、positionRisk/income 对账视图）
  risk.py              RiskGuard：数据陈旧、日亏熔断、FX 跳空告警
  journal.py           SQLite：bars / trades / fills / events 全量落库
  notify.py            Telegram（未配置时降级为日志）
pair_backtest.py       回测 CLI（--mtm-stop 覆盖；输出交易表+权益指标）
run_live.py            实盘 CLI（--testnet 切币安测试网；需 BINANCE_API_KEY/SECRET）
                       无自建 paper 模式：管线演练用 testnet（本策略 symbols 未上线,
                       需换测试 symbols）；策略演练用生产环境 + 极小 leg_notional
scripts/refresh_data.py  刷新 data/ 下的价格与 funding CSV
tests/                 70 个测试：单元 + 对账矩阵 + 平价（引擎 vs 独立 pandas 参考实现逐笔一致，
                       并钉死基线数字：stop2.5→26笔/+16.51%，stop off→24笔/+20.05%）
data/
  skhx_pair_5m.csv     5m 双腿价格 + USDKRW
  funding_kr.csv/us    两腿 funding 结算
  journal.sqlite       实盘 journal（运行后生成）
research/              前期调研存档（币安/HL 股票 perp funding 分析；相对路径未修正）
```

运行:`python3 -m pytest tests/` · `python3 pair_backtest.py` · `python3 run_live.py [--testnet]`

## 多对架构（2026-08-13 起）

每对一个进程/一个 systemd 服务/一份配置/一个 journal,共享账户与代码:

| 服务 | 配置 | 对 | 信号形态 |
|---|---|---|---|
| twopair | deploy/cfg-skhx.json | SKHYNIX/SKHY | 24h 锚+分段 std+MTM 止损(主对 v3) |
| twopair-ewysam | deploy/cfg-ewysam.json | EWY/SAMSUNG | 10d 锚+平坦 std+z 止损 4+14d 超时 |
| twopair-mudram | deploy/cfg-mudram.json | MU/DRAM | 同上 |

部署:`deploy/deploy_all.sh`(或单个 `SERVICE_NAME=.. CONFIG=.. deploy/deploy.sh`);
部署门禁自动断言活跃对之间 symbol 两两不相交。配置字段 kr_symbol/us_symbol
为历史命名 = A 腿/B 腿。新对验证:调研数据灌本引擎,EWY/SAMSUNG 22 笔/82%
胜率复现,MU/DRAM 同量级(见 git 历史)。

状态恢复:仓位以交易所为唯一事实源,每轮(5 分钟)对账——重启恢复只是第 0 轮;
journal 仅补充两个信号空间标量(当日已实现亏损、止损再武装锁存)。
任何时刻 kill 进程,重启即恢复;孤腿/数量漂移/外部手动平仓均自动处理。

## 下一步

1. 生产环境 + 极小名义(如单腿 100 USDT)跑 2~4 周,积累样本外记录与时段标签
2. 对账实盘 vs 回测口径后,VPS(东京)+ systemd 部署,逐步加到目标仓位
3. 手续费恢复收费时重估单笔期望(约 −0.1%/笔)
