# 24h 股票/商品 Perp：除资金费套利之外的机会扫描（总览）

**日期**：2026-08-15/16 **范围**：Binance TradFi perp（149 股票/ETF + 8 商品）、Hyperliquid
xyz（股票/商品/FX），参考价用本机 IBKR（只读）历史数据。
**方法**：6 条独立线并行（各一份子报告，见文末索引），全部用 5m 数据 + 实际费用口径。

## 一句话结论

**"24h perp 是新市场、传统团队还没盯上"这个前提只对了一半**：perp 在闭市时段的价格
发现是有效的（夜间走势被开盘全额确认、跨所同步、与 IBKR 盘外报价的偏差小于对冲腿价差），
所以"方向性/跨场所错价"类的机会基本没有；真正剩下的边来自 **闭市时段流动性薄 + 缺少
自然做市商**——是微观结构 / 流动性提供型的边，容量小、依赖 maker 0 费促销。

## 机会分级

### A. 值得小规模试点（有统计显著性、有实操路径）

| # | 机会 | 关键数字 | 条件/限制 | 子报告 |
|---|---|---|---|---|
| A1 | **死区（00–08 UTC）+ 周末 极端波动被动回归**：EWMA30m ± kσ 挂 maker 单，触碰 EWMA 出 | |z|>4 事件 n=2160，fwd180m 同向 −23 bps（t −7.6），85 名中 70 名为负、2–8 月每月为负；挂单模拟 k=4：DEAD 净 +7.8 bps/笔、31 笔/日、$20k/笔 ≈ $705/日；WKND 净 +14.5 bps、$1.6k/日、72/85 名为正、maxDD $2.7k | 只在 maker 0 费下成立（双 taker 只剩 +3.8）；6–8 月才赚（与扩容重叠，样本短）；赚钱靠中小盘，MU/SNDK/NVDA/TSLA 净为负；死区 5m 成交额仅 RTH 21%，"影线穿越=成交"是上限假设；先用真实盘口验证 4σ 远端成交率 2–4 周；前晚 AH \|收益\|>3% 名字次日不挂 | `offhours_reversal.md` |
| A2 | **ADR/欧股 perp 对母市场实时价的 5 分钟滞后**（ASML 为主） | ASML perp 对 Euronext 实时价 lag+1 相关显著、"母线上根多走>2σ 同向做 perp"：14.8 bps/5m、t 7.4、胜率 73%；basis MR 22 bps/10min（t 5.2，3.8 笔/日）；NVO 边≈费用；港股三线（HK1810 56 bps、TENCENT/HK0700 27–28 bps）潜力最大但只有 17 天样本 | 信号需 IBKR 实时 Euronext/HK 行情；perp 是滞后方、收敛全由 perp 腿完成，因此只需交易 perp；BABA/EWY/EWJ/TSM 零滞后无边 | `adr_homeline_basis.md` |

### A3（08-21 追加）. 动态 cash-and-carry 择时（perp 溢价 + funding 联动）

skfunding 实盘（SK 海力士 carry，07-29→08-21，+$187k/$5.3M）的系统化：KRX 开盘时段
1h 均溢价 ≥30 bps 进（此时 funding 也高——溢价领先 funding）、≤−10 bps 出。80 天回测
SKHYNIX 动态净 1653/1203 bps（低/高摩擦）vs 静态持有 1177，超额一半来自反复卖溢价、
一半来自躲开负 funding 时段；funding 高位有持续性（尾 7d >40% APR → 未来 7d 中位 48%，
横截面 Spearman 0.26–0.50）。勿用 funding 本身做出场（太碎，摩擦吃死）。8 月溢价 regime
已压缩、超额变薄；黄金期是新名字上市头两个月。详见 `dynamic_carry.md`。

### B. 备忘 / 需更多样本再评估

| # | 观察 | 数字 | 子报告 |
|---|---|---|---|
| B1 | 铜/铂/钯 CME 每日休市（17–18 ET）1h 内移动 → 后 3h 反向 | corr −0.47/−0.44/−0.24；\|move\|>10bps 反向 3h 均值 +20/+43/+32 bps，t 2.6–3.1；但周六簿深 ±10bps 仅 $5–80k | `commodity_fx_perps.md` |
| B2 | 原油/BZ perp 资金费持续为负 → 多 perp / 空 CME 期货正 carry | 年化 −24%/−17%（BZ 近 30d −42%）；需 IBKR 期货腿；黄金基差 σ 8 bps 无空间 | `commodity_fx_perps.md` |
| B3 | 宏观日（12:30 UTC）单股 perp 过冲 1.4–1.6× 正常 ES beta，开盘前部分回吐 | 仅 \|ES15\|>15bps 的 6 天里残差回吐 35%（t −4）；样本太小 | `events_calendar.md` |
| B4 | 韩股假日 perp 走势会被 KRX 重开按美盘方向纠正 | 07-17 SAMSUNG/SKHYNIX 假日 −2~4%，同日 EWY +3.3%，次日开盘 1h +342/+466 bps；n=2，08-17 是下一个样本 —— 与现行 SKHYNIX 信号处理直接相关（假日 perp 价格不宜当锚） | `events_calendar.md` |
| B5 | Binance vs HL 资金费差 | HL 系统性贵 4–13% APR（18/22 名字），但 8h 差 std 30–190% APR、结构分量仅 ~5%，回本 5–10 天 | `binance_vs_hl.md` |

### C. 数据否决（不要再试）

- **跨场所 basis 套利（Binance perp vs IBKR 盘前/盘后/隔夜正股）**：perp 恒定高 6–8 bps
  （USDT/USD 稳定币基差），快分量 std 6–12 bps、半衰期 3–6 分钟，p95 16–24 bps；
  \|basis\|>30/50 bps 净 −20/−17 bps；IBKR 盘外价差本身 8.5–13.5 bps。**IBKR OVERNIGHT
  交易所在 00–08 UTC 也有成交**，"perp 独家定价"前提不成立。
- **开盘 snap / 08:00、13:30 机制跳变**：不存在。夜间 perp 走势被盘前全额确认（斜率 1.03、
  r² 0.92），KRX 开盘跳空对 perp 隔夜 β 1.02–1.06、r² 0.91–0.93，两所在四个边界均无系统性跳变。
- **Binance vs HL 同股价差搬砖**：残差 std 3–5 bps、半衰期 ≤2 根、往返费 9.8 bps；韩股腿噪声
  3 倍但事故风险最高。**韩股腿成交级复核（08-18，`hl_kr_spread_mr.md`）：回归现象真
  （>20 bps 偏离 100% 于 24h 内收敛、中位 1–1.5h），但偏离经常在给 BN 韩股 4h 一期、
  单期可达 50 bps 的资金费定价——裸 fade 为负；资金费过滤 + BN maker 后仅 SKHX 一格
  显著（+10.4 bps/笔、t 3.9、0.8 笔/日），容量 $2–5/日，独立策略否决、留观察项。**
- **商品 perp 周末反向**：周末 perp 移动 → CME 重开缺口 β 黄金 0.91 / 白银 1.13 / WTI 0.60，
  期货首 30 分钟还朝 perp 方向补——perp 是周末信息载体，不可反向。
- **24h 财报后漂移**：前 30 分钟完成 AH 位移 79%，DEAD/OPEN/RTH 回归 \|t\|<1.5。
- **美股假日、opex/再平衡、周末→周日 ES 重锚**：无净可交易效应（周末 perp 已走完 ES 缺口 71%）。
- **除息**：Binance 用 ex-date 00:00 UTC 一次性 Special funding 抹平，perp 无提前/滞后计价。
- **杠杆孪生衰减、老牌流动配对**：见 `../pair_trading_research.md`（此前已否决）。

## 结构性事实（对现有策略也有用）

1. Binance 美股指数 = dxFeed/Kaiko/Pyth Pro/Databento/Massive 加权 **÷ USDT/USD**；韩股指数只在
   KRX 主盘 00:00–06:20 UTC 用供应商（Kaiko + Pyth 各 49%、**HL xyz 1%**），其余时段是自家
   盘口 EWMA（2026-05-16 起替代 Fixed）。标记价对指数限 ±5%（盘中）/±3%（周末）。
2. HL xyz 韩股 08:10–20:00 KST 吃外部（含 NXT 盘前/盘后）——它会反映 Binance 忽略的韩股盘前
   打印；闭市内部 EMA τ=30min、discovery bounds ±(1/杠杆)+有限重锚（SKHX ≈19% 封顶）。
   海力士事故 = NXT 盘前 1 股 −29% 打印进 oracle。
3. 死区扎针（>300 bps 且收盘吐回）AH 32 起 / PRE 16 / DEAD 仅 4——扎针风险在盘后不在死区。
4. 周末两所标记价偏离极值 222/116 bps 且不同步：跨所对冲须按单腿 ≥3% 偏离留保证金。

## 建议的下一步（按性价比）

1. A1：写一个只读的 tick/盘口记录器跑 2–4 周，验证 4σ 远端挂单实际成交率，再决定是否上
   $0.4M 保证金的试点（规格见子报告第 7 节）。
2. A2：用 IBKR 实时 Euronext 行情做 ASML 纸面信号 2 周；港股三线等样本到 6 周复核。
3. B4：08-17（韩国光复节补假）观察 SAMSUNG/SKHYNIX perp 假日走势与 08-18 KRX 开盘的关系，
   决定是否在 SKHYNIX 策略里对韩股假日做特殊处理。

## 子报告索引

| 文件 | 线 |
|---|---|
| `offhours_reversal.md` | 死区/周末极端波动回归 + 挂单模拟 + 深度 + HL 对照 |
| `crossvenue_basis.md` | Binance perp vs IBKR 盘前/盘后/隔夜 basis、开盘 snap、除息 |
| `adr_homeline_basis.md` | ADR/ETF perp vs 母市场本线（IBKR）领先滞后、MR、KRX/HK 反向 |
| `commodity_fx_perps.md` | 商品/FX perp 清单、CME 休市窗口、期货 basis、跨所 |
| `binance_vs_hl.md` | 两所指数/标记/资金费机制 + 同股价差/资金费差/领先滞后 |
| `hl_kr_spread_mr.md` | 韩股腿跨所价差均值回归成交级回测（08-18 追加；BN 韩股资金费实为 4h 一期） |
| `dynamic_carry.md` | 动态 cash-and-carry 择时：溢价+funding 联动 vs 静态持有（08-21 追加） |
| `dyn_carry_backtest.md` | 全域动态 funding 套利组合回测：$10M/30d 净 $176k≈21.5% 年化，保守 12–18%；四轮假象修正记录（08-21 追加） |
| `events_calendar.md` | 假日、宏观、财报、opex、周末结构 |
| 脚本 | `scan_offhours_*.py`、`scan_basis_*.py`、`scan_adr_*.py`、`scan_cmdty_*.py`、`scan_hl_*.py`、`scan_events_*.py`、`session_anomaly_scan.py` |
