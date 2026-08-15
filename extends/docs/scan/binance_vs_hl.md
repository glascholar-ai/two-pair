# Binance TradFi 永续 vs Hyperliquid xyz 股票永续：机制对照与跨所价差/资金费实证

**日期**：2026-08-15
**脚本**：`extends/scan_hl_fetch.py`（HL 5m K 线 + 小时资金费 + Binance 标记/指数 K 线抓取，缓存 `extends/data/hl/`）、`extends/scan_hl_spread.py`（价差分段统计、领先滞后回归、资金费差、参考价层）
**中间产物**：`docs/scan/hl_spread_by_segment.csv`、`hl_spread_deviation.csv`、`hl_leadlag_by_segment.csv`、`hl_funding_gap.csv`、`hl_reference_layer.csv`、`hl_boundary_profile.csv`；缓存 `data/hl/xyz_*_5m.parquet`、`xyz_*_funding.parquet`、`bnref_*.parquet`（Binance 标记/指数 5m）、`usdcusdt_5m.parquet`

**结论速览**

1. 两所在母市场闭市时都用"自家订单簿"当参考价，但形式不同：Binance（2026-05-16 起股票合约 Orderbook EWMA 模式）用自家盘口 impact 中价做 EWMA、标记价对指数限 ±5%（盘中/盘前后/夜盘）/ ±3%（周末假日）、EWMA 参数未公开（实测 MU 8-01 00:00 UTC 跳 +1% 时指数约 20 分钟追平）；HL xyz 用部署方 Relayer 的内部 EMA（τ=30 min，由自家盘口 impact 价差推动，单 tick ≤±50 bps、缺口 ≤~9.5%），标记价受 ±(1/最大杠杆) discovery bounds + 有限次重锚约束（SKHX 10x：≈19% 封顶——海力士事故 −17.9% 就是打到这个顶）。两所的美股参考价都覆盖盘前 (08:00 UTC)/盘后/Blue Ocean 夜盘；**韩股腿差异最大**：Binance 只在 KRX 主盘 00:00–06:20 UTC 用供应商，xyz 从 23:10 UTC（NXT 盘前）到 11:00 UTC 都吃外部打印——事故型尾部只在 xyz 侧。
2. **Binance 与 HL 存在一个 ~+10 bps 的恒定水平差**（Binance 高于 HL），20 个美股名字、5 个时段全部成立（RTH 10.2 / AH 10.2 / DEAD 10.6 / PRE 11.1 / WKND 11.1 bps），且在**指数层**就存在（Binance 指数 vs HL oracle 估计值 ≈ 10.3 bps）。**根源是计价币不同**：Binance TradFi 指数的每个成分都以 `//uindex(USDTUSD)` 换算成 USDT，HL xyz 以 USDC/USD 计价，而窗口内 USDC/USDT ≈ 1.0004–1.0011（+4~+11 bps）；扣除同一时刻的 ln(USDCUSDT) 后残差只剩 RTH 1.8 / AH 2.3 / DEAD 2.1 / PRE 2.3 / WKND 3.9 bps。**这是稳定币基差，不是可套的股票错价**（要套它就去做 USDC/USDT）。剔除水平后的快分量 std 仅 3.4–5.1 bps、p99 10–14 bps、半衰期 ≤2 根 5m 棒、|dev|>20 bps 的概率 0.2–0.9%：**美股名字的跨所价差套利在 taker 4+0.9 bps 双边（≈9.8 bps 往返）成本下无空间。**
3. 韩股名字（SKHX/SKHYNIX、SMSN/SAMSUNG）价差 std 10–16 bps、p99 26–47 bps、半衰期 4–13 根、|dev|>20 bps 概率 8–19%，是唯一有"可交易噪声"的一档，但样本只有 13 个交易日、且事件风险最高（HL 海力士 oracle 事故正是这条腿）。
4. 资金费：Binance 8h 一次（00/08/16 UTC）、**~57–67% 的期次恰好为 0**（有死区），全样本均值 ~9% APR；HL xyz 每小时一次、公式经数据反推为 `hourly = [premium + clamp(0.01% − premium, ±0.03%)] / 16`，44% 的小时落在基准 6.25e-6/h（=5.5% APR，多头付）。两所资金费差（BN − HL）近 30 天中位数约 −7% APR（HL 更"贵"），但 8h 差分 std 30–190% APR、lag-1 自相关 0.3–0.5、同号率 0.5–0.7：**只有"HL 基准正费 vs Binance 死区零费"这 ~5% APR 的结构分量是可持续的**，扣掉 9.8 bps 往返成本，回本 5–10 天，收益/事件风险比不划算；个别名字（CBRS +19% APR 同号率 0.75、NBIS −14.5%）是事件驱动的短样本。
5. 领先滞后：同期相关 0.93–0.99；Binance 滞后 1 根对 HL 的预测系数在 20 个名字里 85–100% 为正、闭市时段（DEAD/WKND）中位 t=2.7/3.4，HL 反向中位 t≈0.4：**Binance 在 5m 尺度轻微领先，闭市时段最明显**，但幅度（β≈0.3–0.6 bps/bps，corr≈0.03–0.07）远不足以覆盖费用。
6. 清算不对称：闭市时两所标记价都跟自家盘口走（Binance mark≈last，HL mark≈mid/EMA），跨所对冲头寸在周末会承受**各自独立的**盘口噪声——Binance 标记相对指数的偏离在周末最大达 222 bps（SKHY）、HL premium 最大 116 bps（NBIS）；两所同一小时的偏离方向并不一致。**跨所对冲不能按"净敞口≈0"设置杠杆**，须按单腿裸露 2–3% 的极端标记偏离预留保证金，且 Binance 侧（EWMA 指数滞后 → 资金费尖峰）与 HL 侧（oracle 冻结 → 盘口驱动 mark）失效模式不同。

---

## Part A：机制对照（官方文档 + 公开报道）

（见下文各小节；引用 URL 附在每条之后。未能在官方文档中核实的项已明确标注"未核实"。）

### A.1 Binance USDⓈ-M TradFi 股票永续

**指数价格来源（API `GET /fapi/v1/constituents` 实测，2026-08-15）**

- 美股名字（NVDA/MU/SPCX 等）：`binance_future` 自家合约 0.7% + `databento` 11.0% + `dxfeed NVDA:USLF24` 25.7% + `kaiko KK_RFR_NVDAUSD` 25.7% + `massive FMV.NVDA` 11.0% + `pyth_pro` 25.7%；较新名字（SKHY/CBRS/DRAM）无 databento：dxfeed 28.9% / kaiko 28.9% / pyth_pro 28.9% / massive 12.4% / 自家 0.8%。**每个成分都以 `//uindex(USDTUSD)` 换算成 USDT 计价**——即指数 = USD 价格 ÷ USDT/USD。（dxFeed 的 `USLF24` 是含夜盘的 24 小时美股合并行情；Massive FMV 是"公允市值"合成价。）
- 韩股名字（SKHYNIX/SAMSUNG）：`kaiko KK_RFR_A000660KRW_USD` 49.0% + `pyth_pro SKHYNIX//uindex(USDTKRW)` 49.0% + **`hyperliquid xyz:SKHX` 1.0%** + 自家 1.0%。Binance 韩股指数里挂着 HL xyz 作为 1% 成分（反向依赖：HL oracle 出问题会以 1% 权重进 Binance 指数）。
- 官方 FAQ 只说"第三方数据商加权平均、每秒更新"，不点名供应商：https://www.binance.com/en/support/faq/detail/fe7dcdf24f1943d98b368f5f9f744398 ；上市公告：https://www.binance.com/en/support/announcement/detail/ecf7318c0d434c339e80878588e700d0

**时段模式**（同一 FAQ）：Regular（正常时段，供应商加权、1s 更新）→ Fast-Decay EWMA（盘前/盘后）→ Slow-Decay EWMA（夜盘）→ **Orderbook EWMA**（周末/假日/闭市：以自家盘口 Impact Bid/Ask 中价为指数、再做 EWMA、且"指数移动幅度受限"）；模式切换在 1 分钟内线性混合。**Orderbook EWMA 于 2026-05-16 00:00 UTC 起替代原 Fixed 模式用于股票合约**（商品合约 2026-05-08 21:00 UTC 起：https://www.binance.com/en/support/announcement/detail/d29b59c26a914c71b95e864369a6bfb0 ）。**EWMA 半衰期、移动限幅、impact notional 均未公开**（未核实）；实测：MU 2026-08-01 00:00 UTC 成交跳 +97 bps 时指数 ~20 分钟追平，时间常数量级 5–10 分钟。
- 美股：盘前/盘后/夜盘用供应商数据（Blue Ocean 是否直接入源官方未说明；但 dxFeed USLF24 含 24h 行情，且实测两所在 00:00 UTC 都出现波动率抬升）。
- 韩股：**只有 KRX 主盘 09:00–15:20 KST（00:00–06:20 UTC）走 Regular**，其余时间（含 KRX 盘前 08:00–09:00 KST、盘后至 20:00 KST、NXT）一律 Orderbook EWMA——即 Binance 韩股指数**不吃**盘前/盘后打印；港股 09:30–12:00 & 13:00–16:00 HKT。
- 学院文章佐证：https://www.binance.com/en/academy/articles/how-to-trade-stock-perpetual-contracts-on-binance

**标记价**：正常时段 = 中位数(Price1 = 指数×(1+上期资金费×剩余时间比), Price2 = 指数 + 30s 移动平均的盘口中价溢价, 最新成交价)；闭市时段"Futures Last Price 的 EWMA 平滑"。**标记价相对指数的偏离上限：股票 ±5%（正常/盘前/盘后/夜盘）、±3%（周末/假日）**；分红/公司行动窗口收紧到 1%。https://www.binance.com/en/support/faq/detail/360033525071

**资金费**：8h 一次（00/08/16 UTC）、上限 ±2.00%/期、利率项 0%、**触顶也不缩短周期**（豁免于 8.1 条款）；24/7 收取，闭市时溢价对自参照指数计算 → 结构上被压向 0（实测 57–67% 期次恰为 0）。https://www.binance.com/en/support/announcement/detail/d0833e4ae9b542be90dbf3fe1c960c53

**公司行动**：拆股/并购/分拆"另行公告"（无常设方法学；2026 年拆股公告未检索到，未核实）。**分红**有专门方法学：一次性特别资金费（空付多）Rate = −D/M（现金）、−r/(1+r)（股票股利）、**不受 ±2% 上限约束**；美股流程：除息日前一天 15:30 ET 起→16:00 ET 起资金费改 1h→19:30 ET 只减仓→标记偏离带收紧到 1%→**20:00 ET 执行特别结算**；韩股除息日 08:00 KST、港股 09:01 HKT。https://www.binance.com/en/support/faq/detail/7ced719b5e9a4859a1864c2fe657309f 。**交易停牌（LULD/新闻停牌）无公开政策**（未核实）。

**费用/杠杆/账户**：2026-03-31 02:00 UTC 起"直至另行通知"：TradFi perp maker 0%（全档）、taker 8 折（普通–VIP3，即 0.05%→0.04%）/5 折（VIP4–9），BNB 再 9 折。https://www.binance.com/en/support/announcement/detail/a4c3f1957f2b4e69902985154235c3b1 。杠杆：早期 10x，新上市 25x，韩股 20x 起后提至 50x；合约发行方 Nest Exchange Limited（ADGM/FSRA）。**PM 账户资格、单标的持仓上限、价格保护参数：文档未找到（未核实；本项目 PM 实盘已在跑是唯一证据）。**

### A.2 Hyperliquid HIP-3 股票永续（trade[XYZ]，dex "xyz"）

**Oracle（=资金费参考 + 标记价输入）**：由 xyz Relayer 每 ~3s 推送，"来自多家场所与机构数据商"，**官方不点名**（第三方指南称含 Pyth，未在官方文档核实）。https://docs.trade.xyz/perp-mechanics/oracle-price ；https://docs.trade.xyz/perp-mechanics/overview
- 美股外部覆盖 24/5：周日 20:00 ET → 周五 20:00 ET；盘前 04:00–09:30、主盘 09:30–16:00、盘后 16:00–20:00、**夜盘 20:00–04:00 ET 明确由 Blue Ocean ATS 提供**。https://docs.trade.xyz/asset-directory/stocks/us.md
- 韩股（SKHX/SMSN/HYUNDAI/EWY）：外部报价覆盖盘前 08:10–08:50、主盘 09:01–15:30、盘后 15:40–20:00 KST；其余时段与周末用内部定价；外部数据点间隔 >15s 也切内部；KRW 实时换算 USD（SKHX oracle = 000660.KS ÷ USDKRW）。https://docs.trade.xyz/asset-directory/stocks/korea.md
- **内部定价（闭市）**：连续时间 EMA，被自家盘口的 impact 价差推动：IPD = max(P_impactBid − S, 0) − max(S − P_impactAsk, 0)；S_t = βS + (1−β)(S+IPD)，β = exp(−Δt*/τ)，**τ = 30 min**（2025-11-21 从 8h→1h；2026-04-30 1h→30min），单次更新 Δt* ≤ 0.1τ（≤ ~9.5% 缺口）；外部数据恢复时"下一 tick 回到外部价"。https://docs.trade.xyz/changelog/oracle-time-constant-update.md
- "External price"字段在闭市时冻结于外部收盘。https://docs.trade.xyz/perp-mechanics/external-price.md

**标记价**：中位数(oracle；oracle + 150s EMA(mid − oracle)；median(best bid, best ask, last))；**Relayer 每次更新对 oracle/mark 都夹在当前值 ±50 bps 内**。https://docs.trade.xyz/perp-mechanics/mark-price.md
- **Discovery bounds**：mark 被限制在参考价 ±(1/最大杠杆) 内；oracle 触及带宽 ~90% 时参考价重锚，每方向重锚次数有限（SKHX 10x：±10%、1 次重锚 → 极限约 19%；NVDA/TSLA 20x：±5%、2 次 → ≈15.8%），用尽后硬顶"直到外部定价恢复"。https://docs.trade.xyz/perp-mechanics/discovery-bounds.md ；https://docs.trade.xyz/consolidated-resources/specification-index.md

**资金费**：HL 基础：每小时支付 8h 费率的 1/8，premium 每 5s 采样，利率 0.01%/8h，**上限 4%/小时**，HIP-3 premium = impact 中价/oracle − 1，部署方可设 funding multiplier。https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding 。**xyz 自 2025-12-19 起 multiplier = 0.5**：F₈ₕ = 0.5×[avg premium + clamp(0.01% − premium, ±0.05%)]，基准 ≈ 5.5% APR——与本文数据反推的 `hourly = [P + clamp(0.0001−P, ±0.0003)]/16` 一致（clamp 宽度实测 ±0.03%，官方文档写 ±0.05%，以数据为准）。https://docs.trade.xyz/asset-directory/stocks/updates.md 。**周末/闭市照常按内部 oracle 收费**（12/19 更新明言针对"周末价格发现的资金费压力"）。

**费用/保证金/上限/清算**：标准 HIP-3 费 = 验证者市场 2×（tier-0 taker 0.090%/maker 0.030%），HL 与 xyz 五五分；**growth mode 全部费用减 ≥90%：tier-0 taker 0.0090%（0.9 bps）/maker 0.0030%，tier-1(>5M 14d) 0.0080%/0.0024%，顶档 taker 0.0048%、maker 0–0.003%**；growth mode 排除加密 perp、MSTR 类加密载体、黄金。https://docs.trade.xyz/perp-mechanics/fees.md ；https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees 。HIP-3：部署方质押 50 万 HYPE、费率系数 0–300%（growth ≤100%）、跨保证金需验证者批准、质押可 100% 罚没。https://hyperliquid.gitbook.io/hyperliquid-docs/hyperliquid-improvement-proposals-hips/hip-3-builder-deployed-perpetuals 。保证金文档自相矛盾（一页说 xyz 全部 isolated-only，规格表列 SKHX/NVDA/TSLA 为 cross；事故报道证实 SKHX 当时为 cross、Samsung/Hyundai 为 isolated）；抵押品 USDC，oracle 以 USD 计、无 USDC/USD 换算。https://docs.trade.xyz/trading/margin-and-leverage.md 。**OI "对所有 xyz 市场设上限"但数值未列（未核实）；单地址持仓上限未找到（未核实）**。清算用 xyz relayer 标记价，cross 资产权益 < 2/3 维持保证金触发链上后备清算人，MM = 最大杠杆初始保证金的一半（1.25–16.7%）。https://docs.trade.xyz/risk-and-margining/liquidation-mechanics.md
- **公司行动/停牌：xyz 无成文政策**（仅"方法学可能受公司行动影响"；第三方称拆股按比例调整 oracle，未核实）。→ 除息日 xyz perp 会随现货跳空，而 Binance 用特别资金费抹平——**同一名字除息日两所会拉出一段"分红价差"**（Binance 侧 20:00 ET 一次性结算，xyz 侧次日开盘跳空）。

**SK 海力士事故（xyz:SKHX，2026-07-27 23:01 UTC = 7/28 08:01 KST，NXT 盘前开盘）**：NextTrade 上一笔 1 股 1,272,000 KRW 的打印（较前收 1,785,000/1,816,000 低 ~29–30%）被"多家独立数据商"转发进 oracle；SKHX 标记 $1,127.90 → $917.25（−17.9%，被 10% 带宽 + 1 次重锚封顶于 ~19%）；多头清算 ~$57.4M、~960 账户、已实现亏损 ~$17.3M，空头 ADL ~$10.8M/~100 账户。https://www.coindesk.com/markets/2026/07/29/company-behind-ai-trade-that-caused-usd60-million-crypto-liquidations-to-cover-all-losses ；https://finance.yahoo.com/markets/crypto/articles/hyperliquid-explains-57-million-sk-114320781.html 。Trade.xyz 7/29 表态"oracle 按规格运作"、一次性酌情全额补偿、将"重审外部场所假设、加大自家盘口权重、增加罕见打印过滤"；Hyperliquid 官方："无许可链，各团队自行部署运营市场"。https://www.cryptotimes.io/2026/07/29/trade-xyz-to-cover-sk-hynix-perp-losses-while-insisting-its-oracle-worked/ 。**成文修复只有一条**（changelog 7/28）：韩股外部覆盖窗口收窄——盘前起点 08:00→08:10 KST、主盘 09:00:30→09:01 KST；过滤参数未公开（未核实）。https://docs.trade.xyz/changelog/market-parameters/7-28.md
- 注意 Binance 侧当时不受影响：Binance 韩股指数只在 KRX 主盘走供应商，盘前 08:00–09:00 KST 是 Orderbook EWMA（不吃 NXT 打印）；且 xyz 只占 Binance 指数 1%。

**其他 HIP-3 股票 dex**：`perpDexs` 返回 xyz / flx (Felix) / vntl (Ventuals) / hyna / km & mkts (Markets by Kinetiq) / abcd / cash (dreamcash) / para (Paragon)；仅 Felix 确认通过 Ondo 集成上线部分美股，其余 oracle 细节未核实（检索预算耗尽）。https://hyperliquidguide.com/guides/trading/hyperliquid-xyz-explained

### A.3 时段边界上"可预测"的机制行为（综合）

| 边界 (UTC) | Binance | HL xyz | 含义 |
|---|---|---|---|
| 08:00（美股盘前 04:00 ET） | 供应商 Fast-Decay EWMA 接管（dxFeed USLF24/pyth 等含盘前） | 外部盘前报价接管 | 两所同步吸收盘前，实测 |r| 同步抬升，价差无跳变 |
| 13:30（RTH 开） | Regular 模式 | 主盘外部价 | 两所同步；开盘 5m |r| 70 bps 但价差 |Δ| 仅升至 5.7 bps |
| 20:00 / 00:00（盘后/夜盘） | Fast/Slow-Decay EWMA | 盘后 → Blue Ocean 夜盘 | 都有数据源，无跳空 |
| 周五 00:00 UTC（周六）→ 周一 00:00 UTC（周日 20:00 ET） | 周末 Orderbook EWMA，mark 限 ±3% | 内部 EMA τ=30min，mark 限 ±(1/杠杆) 带宽 + 重锚 | 两条独立盘口过程；周日 20:00 ET 一起"回到外部价"——但实测 3 个周末价差均值 6.8–14.8 bps（≈平时），**没有系统性回吐** |
| 韩股 23:10 UTC（NXT 盘前）/ 00:00 / 06:20–06:30 | 00:00–06:20 UTC 才走供应商，其余自家盘口 | 23:10 起吃外部盘前、盘后到 11:00 UTC | **xyz 反映 Binance 不反映的盘前/盘后打印** → 事故型尾部只在 xyz 侧；正常时段实测未产生稳定价差 |
| 除息日 | 20:00 ET 一次性特别资金费（不设上限） | 无成文规则 → 现货跳空 | 分红价差 = D/M，可预知、单向、跨所 |
| 资金费 | 8h、±2%、死区宽（57–67% 为 0） | 1h、0.5×、基准 5.5% APR | 结构差 ~5% APR（HL 多头付得多），见 B.4 |

---

## Part B：实证（2026-07-28 17:20 → 2026-08-15 05:50 UTC，5m）

### B.0 数据与口径

- HL `candleSnapshot` 只保留最近 ~5000 根 5m 棒 → 有效重叠窗口 **17.5 天**（含 3 个周末、13 个美股交易日、无美股假日）。资金费历史可回溯到 2026-03-01（4014 小时）。
- 名字选择：HL xyz 24h 名义成交量前 22 且在 Binance 有对应合约者：SNDK、SPCX、SKHX→SKHYNIX、DRAM、MU、SKHY、INTC、TSLA、NBIS、SMSN→SAMSUNG、NVDA、GOOGL、EWY、CRCL、AMD、CBRS、MSTR、MRVL、AAPL、MSFT、COIN、SOXL（SOXL 在 HL 只有 8 天）。
- 价差 `spread = ln(BN_close/HL_close)`（bps），只统计两所该 5m 都有成交的棒（美股名字 >99.9%）。
- 时段（UTC）：RTH 13:30–20:00 / AH 20:00–24:00 / DEAD 00:00–08:00 / PRE 08:00–13:30 / WKND 周六日；韩股名字改用 KRX 00:00–06:30 / KR_AH 06:30–09:00 / KR_DEAD 09:00–23:00 / KR_PRE 23:00–24:00。
- 参考价层：Binance `markPriceKlines`/`indexPriceKlines` 5m；HL 小时 `premium` 字段（mid vs oracle）反推 oracle ≈ HL_close/(1+premium)。

### B.1 跨所价差按时段（20 个美股名字，按棒数加权）

| 时段 | 棒数 | 均值 bps | std | 中位数\|s\| | p99\|s\| | 半衰期(5m棒) | P(\|s\|>20) | P(>50) | P(>100) | >20bps 收敛中位分钟 |
|---|---|---|---|---|---|---|---|---|---|---|
| RTH | 20049 | 10.2 | 5.7 | 10.2 | 23.3 | 0.8 | 3.7% | 0.05% | 0.02% | 20 |
| AH | 12921 | 10.2 | 4.6 | 10.2 | 21.1 | 1.8 | 2.8% | 0.01% | 0 | 58 |
| DEAD | 24045 | 10.6 | 4.7 | 10.6 | 21.9 | 1.1 | 3.7% | 0.01% | 0 | 31 |
| PRE | 16653 | 11.1 | 4.3 | 10.9 | 21.6 | 1.1 | 3.5% | 0.01% | 0 | 30 |
| WKND | 22694 | 11.1 | 5.5 | 10.9 | 21.8 | 2.3 | 6.4% | 0.01% | 0.01% | 78 |

- **水平差 ~10 bps 恒定为正**（Binance 高）：逐名字 RTH 均值从 SNDK 7.4 到 CBRS 16.3，20 个名字无一为负；同一名字在 5 个时段的均值差异 <2 bps；逐日看它是一个以天为尺度缓慢漂移的水平（NVDA 7-29 至 8-14：13.9 → 5.4 → 7.4 bps），**不随开收盘切换**，而与 USDC/USDT 同步（USDCUSDT 日收盘 7-28 1.00105 → 8-07 1.00041 → 8-15 1.00093）。
- **USDC/USDT 调整后**（`spread_adj = ln(BN/HL) − ln(USDCUSDT)`）：RTH 均值 1.8、AH 2.3、DEAD 2.1、PRE 2.3、WKND 3.9 bps；逐名字 RTH 调整后均值 −1.0（SNDK）~ +3.6（AAPL）bps，CBRS 8.0 bps 是唯一例外（HL 侧持续负资金费 −13% APR，与其溢价方向一致）。剩余 ~2 bps 与 HL 基准正资金费（多头付 5.5% APR → HL perp 略贴水）量级相符。
- 参考价层证实这是指数/oracle 之差：`ln(BN_index / HL_oracle_est)` 分时段均值 RTH 10.3 / AH 9.3 / DEAD 9.7 / PRE 11.0 / WKND 10.6 bps；2026-08-15（周六）实时快照 NVDA +9.2、TSLA +12.1、AAPL +13.0、MU +6.4、SKHX +18.9 bps。两所各自 perp 对自家参考价的偏离都很小（Binance mark premium 均值 0.7–2.2 bps，HL premium 均值 1.0–1.8 bps）。
- 快分量（价差 − 前 24h 滚动中位数）：std RTH 5.1 / AH 4.4 / DEAD 4.2 / PRE 3.7 / WKND 3.4 bps；p99 10–14 bps；半衰期 0.7–1.8 根；|dev|>20 bps 概率 0.2–0.9%，>50 bps 为 0；越过 20 bps 后回到 10 bps 以内的中位时间 5 分钟（一根棒）。**没有可套的时段结构。**
- 边界剖面（30 分钟槽，工作日）：|Δspread| 全天 2.4–4.1 bps，仅 13:30–14:30 升到 5.2–5.7 bps（两所各自 |收益| 同步升至 70 bps/5m）；`mean Δspread` 每槽 |·|<0.25 bps → **在 08:00 / 13:30 / 20:00 / 00:00 边界均无系统性跳变**。两所都在 08:00 UTC（盘前）和 00:00 UTC（Blue Ocean 夜盘开门）出现同步的波动率抬升（|r| 23.6 / 25.6 bps vs 相邻槽 9–20），说明两所参考价都吸收了盘前与夜盘打印。

### B.2 韩股名字（SKHX/SKHYNIX、SMSN/SAMSUNG）

| 名字 | 时段 | 棒数 | 均值 bps | std | p99\|s\| | 半衰期 | P(\|dev\|>20) | dev std |
|---|---|---|---|---|---|---|---|---|
| SKHX | KRX | 1014 | 10.0 | 11.4 | — | 6.2 | 10.6% | 12.1 |
| SKHX | KR_AH | 390 | 7.5 | 15.1 | — | 10.5 | 18.7% | 16.3 |
| SKHX | KR_DEAD | 2233 | 5.5 | 16.7 | 47 | 12.7 | 11.7% | 13.2 |
| SKHX | WKND | 1218 | −8.1 | 10.7 | — | 4.9 | 0% | 4.3 |
| SMSN | KRX | 1014 | 2.4 | 9.3 | — | 4.2 | 8.4% | 10.0 |
| SMSN | KR_DEAD | 2225 | −2.4 | 14.2 | 40 | 7.0 | 9.2% | 11.9 |
| SMSN | WKND | 1175 | −17.0 | 8.8 | — | 5.3 | 2.2% | 7.1 |

- 韩股名字的价差噪声是美股名字的 3 倍，且以日为尺度换向（SKHX 逐日 KRX 时段均值 −4 ~ +26 bps）；周末均值为负是 8-01/02 那个周末（−18/−26 bps）主导，8-15 周末为 +16/0，**不是稳定的周末结构**。
- 参考价层：Binance SKHYNIX 标记相对指数在 KRX 开盘时段均值 +20–27 bps（Binance perp 在母市场开盘时持续溢价，对应其 41% APR 资金费）；闭市时段 Binance/HL 偏离极值 247/182 bps——韩股腿闭市时的盘口噪声是美股腿的 2–3 倍。

### B.3 领先滞后（5m，r_hl[t] ~ r_bn[t−1] + r_hl[t−1]，及镜像；HC0 t 值）

| 时段 | n | corr(bn_lag, hl) | corr(hl_lag, bn) | 同期 corr | β_bn→hl | β_hl→bn | 20 名字中 t>2 个数 (bn 领先 / hl 领先) | β_bn>0 占比 | 中位 t (bn / hl) |
|---|---|---|---|---|---|---|---|---|---|
| RTH | 19991 | 0.028 | 0.016 | 0.988 | 0.60 | −0.10 | 7 / 0 | 95% | 1.6 / 0.2 |
| AH | 12721 | 0.069 | 0.056 | 0.976 | 0.34 | 0.19 | 5 / 1 | 85% | 1.1 / 0.5 |
| DEAD | 23722 | −0.002 | −0.039 | 0.962 | 0.28 | 0.14 | 12 / 0 | 85% | 2.7 / 0.4 |
| PRE | 16605 | 0.065 | 0.051 | 0.982 | 0.34 | 0.17 | 6 / 1 | 90% | 1.4 / 0.7 |
| WKND | 21143 | 0.050 | 0.033 | 0.928 | 0.38 | 0.07 | 12 / 3 | 100% | 3.4 / 0.5 |

**裁决**：Binance 轻微领先，闭市时段（DEAD/WKND）最显著（12/20 名字 t>2），RTH 里两所几乎同步（母市场同时驱动两边）。经济意义：β≈0.3–0.6 × 上一根 Binance 收益（典型 |r| 10–25 bps）→ 可预测部分 3–10 bps，减去 HL taker 0.9 bps 后名义为正，但 5m 采样、单边成交假设下不可信；且**领先来自 Binance 成交量/参与者更多，不是机制性延迟**（HL oracle 在闭市时不更新，价格发现全靠盘口）。

### B.4 资金费差与跨所 carry

**两所资金费机制（数据反推，文档核对见 Part A）**

- Binance：8h（00/08/16 UTC）；样本 22 名字 2026-06 起 4434 期，**57–67% 期次 = 0**（死区），周末 78% 为 0；均值 9.8% APR（工作日）/ 4.8%（周末），|rate| 极值 0.52%/8h（未触 ±2% 上限）。
- HL xyz：每小时；`hourly = [P + clamp(0.0001 − P, −0.0003, +0.0003)] / 16`（P = 小时 premium；拟合斜率 0.0625 = 1/16，基准 0.0001/16 = 6.25e-6/h = **5.48% APR**，|P| ≤ 0.02–0.04% 时恒等于基准，占 44% 小时；相当于"8h 利率 0.01% + 溢价、再折半"）。样本极值 4.3e-4/h。分时段：AH 8.9 / DEAD 9.5 / PRE 11.1 / RTH 9.5 / WKND 8.2% APR——**闭市时段照常收费，且 premium 绝对值反而更大（WKND |P| 8.8 bps vs RTH 5.8）**。

**年化差 BN − HL（8h 桶对齐，公共窗口 8–164 天）**

| 名字 | 天数 | BN APR | HL APR | 差 APR | 近 30d 差 | 8h 差 std (APR) | ρ1 | 同号率 | 方向（做多便宜所） | 回本天数 taker / maker |
|---|---|---|---|---|---|---|---|---|---|---|
| CBRS | 87 | 5.7 | −13.2 | +18.9 | +21.8 | 37 | 0.07 | 0.75 | 多 HL / 空 BN | 1.9 / 0.3 |
| NBIS | 66 | 10.3 | 24.8 | −14.5 | −6.3 | 46 | 0.23 | 0.76 | 多 BN / 空 HL | 2.5 / 0.5 |
| SKHX | 104 | 41.0 | 49.9 | −8.9 | −20.4 | 128 | −0.12 | 0.58 | 多 BN / 空 HL | 4.0 / 0.7 |
| EWY | 152 | −4.5 | 4.4 | −8.9 | −0.1 | 108 | 0.42 | 0.59 | 多 BN / 空 HL | 4.0 / 0.7 |
| COIN | 164 | −1.4 | 6.2 | −7.6 | −9.6 | 167 | 0.30 | 0.69 | 多 BN / 空 HL | 4.7 / 0.9 |
| SMSN | 106 | 25.4 | 32.6 | −7.3 | −15.4 | 112 | −0.06 | 0.59 | 多 BN / 空 HL | 4.9 / 0.9 |
| SNDK | 128 | 13.4 | 7.1 | +6.3 | −3.5 | 73 | 0.45 | 0.53 | 多 HL / 空 BN | 5.7 / 1.0 |
| MRVL | 94 | 14.9 | 20.9 | −5.9 | −9.8 | 34 | −0.05 | 0.71 | 多 BN / 空 HL | 6.0 / 1.1 |
| MU | 130 | 20.3 | 15.2 | +5.1 | −6.3 | 77 | 0.44 | 0.42 | 多 HL / 空 BN | 7.0 / 1.3 |
| INTC | 164 | 5.3 | 9.8 | −4.5 | −13.0 | 91 | 0.37 | 0.64 | 多 BN / 空 HL | 7.9 / 1.5 |
| AAPL | 141 | 2.7 | −1.5 | +4.2 | +0.1 | 26 | 0.28 | 0.51 | 多 HL / 空 BN | 8.5 / 1.6 |
| NVDA | 143 | 6.1 | 10.2 | −4.1 | −7.1 | 83 | 0.49 | 0.64 | 多 BN / 空 HL | 8.7 / 1.6 |
| AMD | 100 | 12.5 | 8.5 | +4.0 | −6.9 | 38 | 0.45 | 0.35 | 多 HL / 空 BN | 9.0 / 1.7 |
| TSLA | 164 | 0.9 | 3.9 | −3.0 | −7.2 | 80 | 0.40 | 0.53 | 多 BN / 空 HL | 12.0 / 2.2 |
| MSTR | 164 | 1.8 | 5.0 | −3.3 | −11.1 | 188 | 0.34 | 0.60 | 多 BN / 空 HL | 10.9 / 2.0 |
| CRCL | 164 | 15.9 | 12.5 | +3.4 | −8.1 | 146 | 0.34 | 0.46 | 多 HL / 空 BN | 10.5 / 1.9 |
| GOOGL/MSFT/SPCX/DRAM/SKHY/SOXL | — | — | — | \|差\|<2 | −8~+3 | — | — | — | — | >20 |

（回本天数 = 往返费用 / 年化差；taker 口径 = Binance 4 bps×2 + HL 0.9 bps×2 = 9.8 bps；maker 口径 = Binance 0 + HL 0.9×2 = 1.8 bps。HL 0.9 bps 取自 growth mode 文档区间 0.45–0.9 bps 上沿，xyz `deployerFeeScale=1.0`。）

**裁决**

- 全样本差的中位数只有 |4–6|% APR，**近 30 天 22 个名字里 18 个为负（HL 比 Binance 贵 4–13% APR）**——这就是"HL 基准 5.5% 正费 + Binance 死区零费"的结构差，方向可预测但幅度小。
- 8h 差分的 std 是均值的 5–30 倍，ρ1 0.3–0.5，同号率 0.5–0.7：碰到事件期（SKHX 128%、MSTR 188%、COIN 167% std）单期就能抹掉一个月的 carry，而且**事件期两所资金费同向飙升，差分反而不稳定**。
- 结论：**多 BN / 空 HL 吃 ~5–7% APR 的结构差在数学上为正，回本 5–10 天，但要承担 (a) 10 bps 水平差以天为尺度 ±5 bps 的漂移（≈ 一周 carry），(b) HL oracle 事故型尾部（海力士事件 −17.9%），(c) 双所保证金 2× 资金占用；不值得作为独立策略上线。**唯一有意义的用法是：已经持有 Binance 多头/空头时，把对冲腿放到资金费更有利的一侧（例如 CBRS 空 HL 而非空 BN 可多赚 ~19% APR，若其 −13% APR 的 HL 负费持续）。

### B.5 闭市时段的 oracle/mark 差异与清算不对称

- Binance 闭市指数 = 自家成交 EWMA：MU 2026-08-01 00:00 UTC 成交从 814 跳到 822（+97 bps），指数 814.5 → 815.7 → 816.6 → 817.0，20 分钟后才追平 → **标记价（≈last）与指数偏离 97/72/31/20 bps 递减**。这种偏离直接进入 Binance 的 premium → 资金费；对清算而言 Binance 用标记价，标记价跟盘口，所以清算风险来自盘口而非指数滞后。
- Binance mark 相对 index 的 |偏离| 极值：RTH 中位 14 / 最大 39 bps；AH 26/69；PRE 29/72；DEAD 18/77；**WKND 66/222 bps**（SKHY、NBIS）。HL premium |·| 极值：RTH 10/23；WKND 66/116 bps。两所周末的偏离量级相当，但**同一小时并不同向**（idx_gap 的周末 std 15.7 bps 是 RTH 的 3.4 倍）。
- 韩股腿：Binance SKHYNIX/SAMSUNG 的标记偏离 >50 bps 的棒分别有 641/548 根（其余名字 1–89 根），闭市极值 247/260 bps；HL 端 182/191 bps。
- **对跨所对冲的含义**：(1) 两腿的标记价在闭市时是两条独立的、由各自薄盘口驱动的过程，"净敞口≈0"不能换成低保证金——单腿须按 ≥3% 的标记偏离（周末极值 2.2% ×1.5 缓冲）+ 母市场跳空预留；(2) HL 侧的失效模式是 oracle 冻结/坏打印（海力士事件），Binance 侧是 EWMA 追价滞后 → 资金费尖峰；(3) 韩股腿的闭市噪声是美股腿的 3 倍，且 HL SKHX 的清算历史（$57M）就发生在母市场盘前——若必须跨所做韩股腿，应把 HL 腿放在保证金更宽裕的一侧或干脆只用 Binance。

### B.6 局限

- HL 5m 只有 17.5 天、3 个周末，无美股假日样本；韩股名字只有 13 个 KRX 交易日。
- 用 5m 收盘价近似成交，未建模盘口深度；HL 缓存的是成交 K 线，无历史 oracle/mark（用小时 premium 反推）。
- 费用取当前促销（Binance maker 0/taker 4 bps；HL growth mode），取消后美股名字的所有结论只会更保守。
