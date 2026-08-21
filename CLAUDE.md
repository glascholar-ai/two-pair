# 项目背景

本仓库是 **two-pair 策略**开发与运行目录：币安 SKHYNIXUSDT（韩国线）vs
SKHYUSDT（美国 ADS 线）配对均值回归。基线 v3（`pair_backtest.py` 可复现）：
FX 调整 log 比价、24h 滚动锚（288×5m，锚窗口须为 24h 整数倍——比价有日内
季节性）、5 时段分段 std（窗口 300，安全区 300–450）、z 进 2 / 出 0.5 /
24h 超时、MTM 止损 2.5%（带再武装）。样本 2026-07-10~08-01：26 笔、
+16.51%、胜率 77%、maxDD −4.18%（单腿名义口径，持仓时总敞口 2×）。

**已用数据裁决过的设计结论（不要重新试错）：**

- 5m 信号采样优于 1m/实时：采样延迟造成的进场过冲（均值 2.56σ vs 2.33σ）
  是均值回归的隐性优势；实时数据只用于风控与执行层
- USDKRW 调整必须做（2026-07 韩元贬 4%，不调会污染信号）
- 时段硬规则（跳过美股盘中、开盘催化剂离场）不稳健，降级为观察项；
  分段 std 是其原则性替代
- 仓位不随 z 加深而加仓（z>4 的 episode 均值为负——深 z 是锚断裂证据）；
  z>4.5 只做告警
- 止盈全档位有害（收敛退出本身是自适应止盈，赢单峰值回吐仅 0.06%）；
  `mtm_take_profit_pct` 保留但默认 0
- 杠杆 λ=单腿名义/资金 ≤2，币安全仓模式，保证金缓冲 ≥3× 历史 MDD
- 改配置名义额是常规操作，接管仓位不许拍平重开：超配→reduce-only 削减到
  配置；欠配→原样接管跑完，下次进场用新额度；MTM 止损分母一律用仓位
  **实际**单腿名义（尺寸差异是资金管理偏好，不是安全问题）
- 止盈复核（2026-08-17）：固定/追踪止盈在两新对回测中仍无净收益（回测
  回吐结构近乎二元：收敛单零回吐 15/16、崩溃单 1/16）；但回测仅 22 笔/
  8 周，条件概率是 0/8 级别的点估计（95% CI 上界 ~31%），且实盘 ewysam
  一笔已出现回测未采样的"慢磨回吐"模式（持仓 84h > 样本最长 47h，回吐
  172bps）。追踪止盈已实现为配置项 trail_arm_pct/trail_gap_pct（默认关，
  最稳健档 arm=2.5/gap=1.0）。**预注册升级规则：实盘累计 15 笔后，若
  "峰值≥均值"组实测 P(回吐≥100bps) ≥20%，启用追踪止盈**——由数据触发，
  不由单笔交易触发。
- 多对扩展（2026-08-13）：EWY/SAMSUNG、MU/DRAM 上线（各 50k/腿），用调研
  验证的形态：10d 平坦 std、z 止损 4、14d 超时、无 FX 项、无 MTM 叠加。
  **分段 std 在慢对上有害**（时段边界换分母 × 慢残差 = z 幻影跳变，
  MU/DRAM 变体回测 −9011bps/586 次假止损）——分段专属于快回归+时段驱动
  错位的结构；MTM 止损叠加在 z 止损之上为小幅负贡献，不加。参数必须
  匹配回归时间尺度（半衰期 67h/172h → 10d 窗口；5d 窗口 MU/DRAM 胜率
  从 88% 崩到 23%）。TSM/EWT 因 EWT 容量($0.7M/日)否决；CRWV/NBIS 因
  7 月集中度否决进观察名单。每对一进程，symbol 不相交是部署硬门禁。

**生产部署**：GCE `instance-two-pair`（东京，静态 IP 34.84.44.188）
`/home/luna/trading`，systemd `twopair.service`，PM 账户（papi 端点，注意
papi 有 ~1s 读后写延迟、无 countdownCancelAll）、密钥在 GCP Secret Manager
（`binance-t32-apikey`/`binance-t32-secret-key`）、Telegram 已配置。
仓位以交易所为唯一事实源（每轮对账），journal.sqlite 仅作研究记录。
用户曾同时运行海力士 cash-and-carry（IBKR 正股 + 币安空 perp，2026-07-29→
08-21 清仓，费后 +$187k/$5.3M，总结 `~/app/skfunding/summary.txt`），与本
策略风险互补；动态化复盘见 `extends/docs/scan/dynamic_carry.md`（溢价+
funding 联动择时优于静态持有），funding 追踪器在 `~/app/stocka/perpfund`。
**funding 套利类策略的用户风险规则：perp 侧杠杆 ≤1×（开仓名义不超过保证
金），不得以更高杠杆假设做资本效率论证**；全域动态回测与 A/B（对冲腿=正股
vs 对面 perp）裁决见 `extends/docs/scan/dyn_carry_backtest.md`。

# 工作规则

- **未经用户明确要求,绝不重新部署或重启交易程序**：包括 deploy.sh、
  `systemctl restart/stop/start twopair`、以及任何会导致 twopair.service
  重启的操作。修复类/功能类改动一律：本地实现 + 测试 + git 提交推送后
  **停下**,向用户说明改动内容与部署必要性,由用户决定是否上线及时机。
  （历史教训：2026-08-02 两次未攒批的部署重启恰逢策略开平仓,重启通知
  与交易通知交织,造成"平仓未完成"的误判。）
- **离线研究不碰远程服务**：回测/参数扫描/分析性改动只在本地跑 + git
  提交推送；每次 restart 都会给用户的 Telegram 推送 "loop starting"
  造成打扰。

# 代码规范

Python 代码必须遵守：

1. **Type hints**：所有函数签名（参数与返回值）必须带类型标注。
2. **长度限制**：单个函数不超过 200 行；单个文件不超过 1000 行。
3. **静态检查零告警**：每次改动 Python 代码后运行 `npx pyright`（配置见
   `pyrightconfig.json`，basic 模式，与 VS Code Pylance 一致），必须保持
   **0 errors / 0 warnings** 才算完成。收窄 pandas/Optional 类型时优先用
   显式判空与 `cast`，禁止用 `# type: ignore` 敷衍（确有第三方库标注缺陷
   时须附一行原因注释）。
4. 其余遵循 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)。
