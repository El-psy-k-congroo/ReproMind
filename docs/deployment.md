# 部署与运维

## 模式与边界

`DEPLOYMENT_MODE=local` 是默认完整模式，无需登录。仅绑定本机或可信内网。面向公众提供完整执行能力前，需要用户认证、数据隔离和操作系统级执行沙箱；确认按钮不能替代沙箱。

`DEPLOYMENT_MODE=demo` 用于受访问口令保护的共享评审。允许项目、材料与复现助手操作；服务端拒绝复现准备、环境安装、实验执行、源码修改和全局模型调度写入。未列入允许列表的新写接口默认拒绝。助手给出的操作建议仍要通过相应 API，不能绕过限制。

当前没有账号注册与用户间数据隔离。共享评审工作区不能上传私密材料。默认最多 12 个项目和 256MB 材料，代理另限制上传、请求频率和并发。此模式适用于受控评审，不是面向任意用户的开放服务。

## 服务器结构

- `/opt/repromind`：完整 Git 跟踪代码和前端构建，无本地实验历史。
- `/opt/repromind-api-venv`：Python 3.11 / 3.12 应用环境。
- `/etc/repromind/repromind.env`：root 所有，600 权限，不在网页根目录。
- `/etc/repromind/review.htpasswd`：访问口令哈希，root:www-data，640 权限。
- `/var/lib/repromind`：repromind 用户所有，保存数据库、上传和项目工作区。
- Nginx 监听 80 / 443；API 仅监听 127.0.0.1:8010。
- systemd 以普通用户运行应用，限制内存、CPU、进程数量和可写目录。

Ubuntu 22.04 自带 Python 3.10，不满足项目要求。应单独安装 3.11 / 3.12，不替换系统 Python。前端可在开发机构建后上传，降低服务器内存需求。

私有配置示例：

```dotenv
APP_ENV=production
APP_HOST=127.0.0.1
APP_PORT=8010
DEPLOYMENT_MODE=demo
DATABASE_URL=sqlite+aiosqlite:////var/lib/repromind/repromind.db
UPLOAD_DIR=/var/lib/repromind/uploads
PAPER_PROJECTS_DIR=/var/lib/repromind/paper-projects
MODEL_ROUTING_PATH=/var/lib/repromind/model-routing.json
MAX_UPLOAD_MB=10
DEMO_MAX_PROJECTS=12
DEMO_MAX_STORAGE_MB=256
LANGGRAPH_CHECKPOINT_URL=
# 再按 README 填写模型提供方和密钥。
```

建立系统用户 `repromind`，准备上述目录和 Python 环境，安装后端 requirements。应用用户不得修改 `/opt/repromind`。将 `deploy/repromind.service` 安装到 `/etc/systemd/system/`：

```bash
systemctl daemon-reload
systemctl enable --now repromind
systemctl status repromind --no-pager
curl --fail http://127.0.0.1:8010/api/health
```

## TLS 与访问口令

先安装 Nginx，配置 80 端口 `/.well-known/acme-challenge/` 的 webroot，并确认公网可达。单独生成随机评审口令和 htpasswd 哈希，不复用 SSH 密码。TLS 完成前不开放需要登录的应用。

替换 `deploy/nginx.conf` 中的 `SERVER_IP`。HTTP 跳转 HTTPS；前端、API、SSE 均有访问口令保护。SSE 禁用缓冲，读取超时 300 秒。写请求超限返回 429。云安全组允许 TCP 80 / 443，不开放 8010。

域名可使用常规域名证书。公网 IP 可使用 Certbot 5.4+ 的 IP 证书能力（见 [Let’s Encrypt 官方说明](https://letsencrypt.org/2026/03/11/shorter-certs-certbot)）：

```bash
/opt/certbot/bin/certbot certonly --webroot -w /var/www/letsencrypt \
  --ip-address SERVER_IP --preferred-profile shortlived
```

先使用 staging 验证挑战，再签发生产证书。IP 证书有效期短，必须安装 `deploy/repromind-cert-renew.service` 和 `.timer`，每日检查两次，续期后 reload Nginx。私钥不进入 Git。

## 更新、验收和回滚

1. 确认活动实验已停止；备份数据库、上传、私有配置和当前代码版本。
2. 部署已验证提交及对应前端构建，保持前后端版本一致。
3. `nginx -t` 后重启服务，检查 systemd 状态与日志。
4. 分别验证 API、TLS、未登录 401、登录后页面、主脚本、前端 API 路径、项目接口和浏览器渲染。
5. 验证 demo 拒绝执行及全局设置写入；确认开机启动和续期 timer。

```bash
systemctl status repromind nginx repromind-cert-renew.timer --no-pager
journalctl -u repromind -n 80 --no-pager
nginx -t
systemctl restart repromind
/opt/certbot/bin/certbot certificates
```

回滚时恢复上一版代码和前端。数据库恢复必须先停止服务，不能覆盖正在使用的记录。不要把私有配置和敏感日志发布到仓库。
