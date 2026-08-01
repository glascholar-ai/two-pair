# 部署（GCE 东京 · e2-small · Debian 12）

前置(已完成的):静态外部 IP + 币安 API key IP 白名单;VM 附加服务账号
`two-pair-server@glascholar.iam.gserviceaccount.com`,access scope 为
"Allow full access to all Cloud APIs";全局 secret:
`binance-t32-apikey`、`binance-t32-secret-key`,SA 已授 secretAccessor。

## 安装步骤

```bash
# 1. 代码与依赖
sudo useradd -r -m -d /opt/twopair -s /usr/sbin/nologin twopair || true
sudo git clone git@github.com:glascholar-ai/two-pair.git /opt/twopair
cd /opt/twopair && sudo -u twopair python3 -m pip install --user -r requirements.txt

# 2. 验证密钥链路(以 twopair 用户)
sudo -u twopair deploy/start.sh --help   # 应打印 run_live 帮助后退出

# 3. systemd
sudo cp deploy/twopair.service /etc/systemd/system/
#    如需 --testnet / --config,编辑 ExecStart 行追加参数
sudo systemctl daemon-reload
sudo systemctl enable --now twopair
journalctl -u twopair -f
```

## 说明

- `start.sh` 启动瞬间从 Secret Manager 取密钥注入进程环境,不落盘、
  不写进 unit 文件;轮换密钥 = 添加新 secret 版本 + `systemctl restart twopair`。
- Telegram:`SECRET_TELEGRAM_TOKEN`(secret 名)与 `TELEGRAM_CHAT_ID` 两个
  环境变量可在 unit 的 `Environment=` 行配置,不设则通知降级为日志。
- 状态查看:`python3 scripts/status.py`(VM 上带 SA 无 Binance key 也只能看
  journal 部分;交易所部分依赖 BINANCE_API_KEY,可用
  `sudo -u twopair bash -c 'source <(deploy/start.sh的取值部分)'` 或直接看币安 UI)。
- 每日 Telegram 摘要缺席 = 循环未运行,先看 `systemctl status twopair`。
