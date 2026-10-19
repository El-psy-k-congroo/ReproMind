# 部署与运维

## 部署模式

- `DEPLOYMENT_MODE=local`：本机或可信内网，无需账号，保留原有项目记录。
- `DEPLOYMENT_MODE=server`：HTTPS 网站，注册登录后使用完整功能。项目、附件、实验、报告、对话、统计与模型偏好按账号隔离。

不再提供共享演示工作区，不使用 Nginx Basic Auth，不按服务器型号禁用下载、环境管理、源码修改或实验入口。没有 GPU 会影响具体实验的设备选择与可运行性，不能虚报训练完成或论文复现成功。

## 账号与数据

密码使用带随机盐的 scrypt 哈希。随机会话令牌仅保存在 HttpOnly / Secure / SameSite Cookie，数据库保存令牌哈希。登录会话可撤销；退出撤销当前会话，改密和找回密码撤销全部旧会话。写请求校验来源，登录与恢复有频率限制。

注册后显示一次恢复密钥；忘记密码时使用邮箱、恢复密钥和新密码完成恢复。密钥成功使用后轮换，不能重复使用。不依赖邮件发送服务，也不提示未实际发送的邮件。恢复密钥丢失需维护者核实身份后人工处理。

新增账号不会继承未分配的历史项目。升级时先备份数据库，再由维护者明确指定历史项目的 owner_id；不能批量分配给每个用户。数据库表为增量创建，不删除旧实验记录。

## Linux 服务结构

- `/opt/repromind`：应用源码与构建后的前端，只读。
- `/opt/repromind-api-venv`：Python 3.11 / 3.12 后端环境。
- `/opt/repromind-conda`：Conda 工具基础安装，只读。论文环境在用户确认后创建。
- `/etc/repromind/repromind.env`：root 所有，600 权限，保存服务私有配置。
- `/var/lib/repromind`：数据库、上传文件、项目目录和独立论文环境。
- Nginx 监听 80 / 443；单个 API 进程监听 127.0.0.1:8010。
- systemd 以 repromind 普通用户运行，限制 CPU、内存、交换空间与进程数。

私有环境配置示例：

```dotenv
APP_ENV=production
APP_HOST=127.0.0.1
APP_PORT=8010
DEPLOYMENT_MODE=server
PUBLIC_ORIGIN=https://你的公网IP或域名
SESSION_SECURE_COOKIE=true
EXECUTION_SANDBOX=true
DATABASE_URL=sqlite+aiosqlite:////var/lib/repromind/repromind.db
UPLOAD_DIR=/var/lib/repromind/uploads
PAPER_PROJECTS_DIR=/var/lib/repromind/paper-projects
MODEL_ROUTING_PATH=/var/lib/repromind/model-routing.json
MAX_UPLOAD_MB=40
LANGGRAPH_CHECKPOINT_URL=
# 再按 README 填写模型配置。
```

Ubuntu 22.04 系统 Python 3.10 不满足要求，应单独安装 3.11 / 3.12，不替换系统 Python。后端安装 requirements，前端执行 npm ci 与 npm run build。建立普通系统用户、数据目录和私有配置后，安装 deploy/repromind.service。

## 执行隔离

公网运行作者代码需要安装 bubblewrap，并确认普通用户可以使用 Linux 用户命名空间。所有外部复现进程通过统一入口运行，保留当前项目和该项目环境的可写绑定，系统工具与基础解释器只读；不绑定其他项目、业务数据库、服务私有配置或宿主进程视图。子进程使用显式白名单环境，不继承 API 密钥。环境探测和移除也使用隔离入口。

Conda 项目环境位于 `/var/lib/repromind/environments/`，包缓存与运行缓存位于项目内。服务器一次调度一个外部执行进程，避免并发安装与实验抢占内存；本地调度行为保持不变。安装包仍需联网。用户上传的数据与项目文件均需定期备份；系统不保证任意第三方代码都能在资源有限的服务器成功复现。

数值比较默认使用 `REPRODUCTION_COMPARISON_MODE=practical`：准确率等比例指标允许 `REPRODUCTION_RATIO_TOLERANCE=0.01`（1 个百分点），其他指标允许 `REPRODUCTION_RELATIVE_TOLERANCE=0.05`（目标值的 5%）。论文报告的不确定性更大时取较宽者；页面和报告同时显示实际分数、差值和判定依据。可设置 `strict` 仅使用论文报告的不确定性。任务准备时保存策略，不改写历史结果；数值达标只覆盖本次选择的实验范围，不代表全部论文主张已验证。Windows 与 Linux 使用相同配置项。

## HTTPS 与运维

替换 deploy/nginx.conf 的 SERVER_IP。HTTPS 静态页面公开，业务 API 在应用层校验会话和归属。SSE 禁用缓冲、读取超时 300 秒；上传上限与后端一致。80 端口仅提供证书挑战与 HTTPS 跳转。

IP 证书使用支持 IP 地址的 Certbot 5.4+ 和 shortlived profile，参考 [Let’s Encrypt 官方说明](https://letsencrypt.org/2026/03/11/shorter-certs-certbot)。安装 deploy/repromind-cert-renew.service 和 .timer，每日两次检查，续期后 reload Nginx。

```bash
systemctl daemon-reload
systemctl enable --now repromind nginx repromind-cert-renew.timer
nginx -t
systemctl status repromind --no-pager
curl --fail http://127.0.0.1:8010/api/health
journalctl -u repromind -n 80 --no-pager
```

更新前确认没有活动实验，备份数据库、上传、配置与上一个版本。分别验证前端入口、主脚本、健康接口、前端 API 路径、本人项目接口、另一账号越权请求和浏览器真实渲染。注册、登录、退出、会话失效、密码恢复及改密需要独立验证。构建成功不等于服务已运行。

回滚时停止服务再恢复代码和数据备份，不能覆盖使用中的实验记录。密钥、恢复密钥、会话 Cookie、数据集、模型权重、数据库与运行记录不提交 Git。


### 软件包下载缓慢

先查看环境构建的实时操作日志，区分正在下载、依赖冲突和训练错误。若默认 PyPI 路径持续缓慢，可由部署者在私有环境配置中设置单一的 `PIP_INDEX_URL`，例如阿里云镜像 `https://mirrors.aliyun.com/pypi/simple/`。本地 `.env` 与服务器 systemd 环境均支持此配置；空值沿用 pip 原有设置，作者命令显式指定的索引仍优先。

先通过页面暂停任务，确认外部安装进程已退出，再更新配置并重启服务、继续任务。继续时复用已有项目环境及下载缓存，不需删除环境或改写作者源码。不添加 `--trusted-host`、不关闭 TLS 验证、不放宽作者包版本；镜像缺包或版本冲突仍应保留日志并核对依赖。
