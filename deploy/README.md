# 部署（GCE 东京 · e2-small · Debian 12）

前置(已完成的):静态外部 IP + 币安 API key IP 白名单;VM 附加服务账号
`two-pair-server@glascholar.iam.gserviceaccount.com`,access scope 为
"Allow full access to all Cloud APIs";全局 secret:
`binance-t32-apikey`、`binance-t32-secret-key`,SA 已授 secretAccessor。

## 一键部署(本地执行)

```bash
deploy/deploy.sh                        # 测试门禁 -> rsync -> venv 依赖 -> systemd -> 重启
SERVICE_ARGS="--testnet" deploy/deploy.sh   # 需要传给 run_live.py 的参数
```

目标机 `instance-two-pair`(REMOTE 变量可覆盖),部署到 `/home/luna/trading`
(DEST 可覆盖),以 luna 用户运行,依赖装在 venv(Debian 12 的 PEP 668 限制)。
远程 `data/`(journal)永不被 rsync 触碰,部署不丢研究数据。

首次部署后验证:`ssh instance-two-pair journalctl -u twopair -f`,
应看到 credentials loaded / warmup / 循环启动。手动单测密钥链路:
`ssh instance-two-pair /home/luna/trading/deploy/start.sh --help`。

## 说明

- `start.sh` 启动瞬间从 Secret Manager 取密钥注入进程环境,不落盘、
  不写进 unit 文件;轮换密钥 = 添加新 secret 版本 + `systemctl restart twopair`。
- Telegram:`SECRET_TELEGRAM_TOKEN`(secret 名)与 `TELEGRAM_CHAT_ID` 两个
  环境变量可在 unit 的 `Environment=` 行配置,不设则通知降级为日志。
- 状态查看:`python3 scripts/status.py`(VM 上带 SA 无 Binance key 也只能看
  journal 部分;交易所部分依赖 BINANCE_API_KEY,可用
  `sudo -u twopair bash -c 'source <(deploy/start.sh的取值部分)'` 或直接看币安 UI)。
- 每日 Telegram 摘要缺席 = 循环未运行,先看 `systemctl status twopair`。
