# 部署与运维

## 从运行包在本地部署

运行包由应用源码、已经构建的 `frontend/dist`、启动脚本和无密钥的 `.env.example` 组成，适用于 Windows 与 Linux。它不是包含操作系统和所有依赖的离线镜像。首次安装后端依赖、调用模型以及下载论文资源需要网络；不携带项目数据库、实验、数据集、权重或开发者账号。

1. 安装 Python 3.11 或 3.12，确认 `python --version`。Linux 同时安装该版本的 venv 模块。完整复现需要 Git；论文独立环境需要 Conda。仅启动已构建的工作台不需要 Node.js 或 GPU。
2. 把压缩包解压到一个自己可写的目录。不要在压缩软件中直接运行，也不要覆盖旧部署的数据目录。
3. 首次启动可先把根目录 `.env.example` 复制为 `.env`；配置好下面的模型选项。已有 `.env` 不会被启动脚本覆盖。
4. Windows PowerShell 在解压目录执行 `powershell -ExecutionPolicy Bypass -File .\start-windows.ps1`。这只对本次进程设置执行策略，不修改系统策略。也可直接执行 `py -3.11 start.py`。
5. Linux 在解压目录执行 `bash start-linux.sh`。脚本优先选择 Python 3.11，其次为 3.12；需要指定解释器时执行 `python3.12 start.py`。
6. 首次启动会创建应用专用 `.venv` 并安装 `backend/requirements.txt`，后续依赖未变则直接启动。若已有 `.venv` 版本不符合要求，脚本停止并保留原目录。安装失败会显示错误，修正网络或依赖后可再次启动；不会自动训练或创建论文 Conda 环境。
7. 终端看到 Uvicorn 监听后，浏览器访问 `http://127.0.0.1:8010`。保留该终端；停止工作台用 Ctrl+C，先确保论文任务已经暂停并且外部进程退出。

前端文件由后端同源提供，不需要再开一个前端服务器。如果从 Git 源码部署而没有 `frontend/dist`，先安装 Node.js 22.12+，在 `frontend` 执行 `npm ci` 和 `npm run build`，再运行启动脚本。

## 配置移动云 MoMA

编辑根目录 `.env`，选择 `MODEL_PROVIDER=moma`，填写自己的 `MOMA_API_KEY` 和可用模型标识。`MOMA_BASE_URL` 是公开接口根地址，不是个人密钥。下面只列无密钥模板：

```dotenv
MODEL_PROVIDER=moma
MOMA_API_KEY=填写有效密钥
MOMA_BASE_URL=https://zhenze-huhehaote.cmecloud.cn/v1
MOMA_MODEL_ROUTER=填写账号可用模型
MOMA_MODEL_CHAT=填写账号可用模型
MOMA_MODEL_CODER=填写账号可用模型
MOMA_MODEL_REASONING=填写账号可用模型
DEPLOYMENT_MODE=local
APP_HOST=127.0.0.1
APP_PORT=8010
```

模型接口通过 HTTPS 的兼容对话协议调用。四个能力组可以共用模型，也可分别选用账号可访问的模型。API 提供论文理解、规划、编码与助手等推理能力，不提供本机论文实验的 GPU。模型 API 的地址、账号授权、额度、区域网络和可用模型需要有效。缺少密钥不妨碍启动工作台，但依赖模型的功能不可用；401/403 应检查密钥或授权，429 应检查额度与限流。修改后重启后端，不在浏览器、源码、终端截图或日志中暴露密钥。

评审专用私有配置如由参赛者单独提供，只把其中 `MODEL_PROVIDER` 与 `MOMA_*` 项复制到本地 `.env`；不需要任何服务器登录信息。公开仓库和运行程序压缩包均不带该配置。

## 本地数据与 Conda

默认 `DATABASE_URL=sqlite+aiosqlite:///./data/repromind.db`。启动脚本固定以 `backend` 为工作目录，因此数据库在 `backend/data/repromind.db`，附件在 `backend/data/uploads`，源码、日志及结果在 `backend/data/paper-projects`。这些内容属于当前部署实例，与其他网站实例无关。首次启动会建立目录及表结构，无需创建远程数据库或开 SSH 隧道。

Windows 绝对路径示例：

```dotenv
DATABASE_URL=sqlite+aiosqlite:///C:/ReproMindData/repromind.db
UPLOAD_DIR=C:/ReproMindData/uploads
PAPER_PROJECTS_DIR=C:/ReproMindData/paper-projects
MODEL_ROUTING_PATH=C:/ReproMindData/model-routing.json
```

Linux 绝对路径示例：

```dotenv
DATABASE_URL=sqlite+aiosqlite:////home/your-user/ReproMindData/repromind.db
UPLOAD_DIR=/home/your-user/ReproMindData/uploads
PAPER_PROJECTS_DIR=/home/your-user/ReproMindData/paper-projects
MODEL_ROUTING_PATH=/home/your-user/ReproMindData/model-routing.json
```

将 `your-user` 换为本机用户，保证目录可写。路径改变不会自动复制旧数据；已有项目应先停任务并备份，再迁移数据库与项目文件。SQLite 默认启用 WAL；备份应使用 SQLite 一致性备份接口，或停止应用后连同相关文件完整备份，不能在写入时只复制主文件。PostgreSQL 为可选业务存储，需另行部署并迁移已有数据；LangGraph 检查点适配与业务数据库不同，当前主流程无需单独启用它。

完整作者实验需要 Miniforge/Conda。请从其官方渠道安装并核对安装包，在启动工作台的终端确认 `conda --version`。若没有加入 PATH，可设置 `CONDA_EXE` 为实际 conda 可执行文件，例如 Windows 的 `C:/Miniforge3/Scripts/conda.exe` 或 Linux 的 `/home/your-user/miniforge3/bin/conda`。Windows PowerShell 用 `$env:CONDA_EXE='实际路径'`，Linux 用 `export CONDA_EXE='实际路径'`，随后在同一个终端启动工作台。

工作台环境与作者实验环境分开。系统读取作者入口和依赖，再检查本机系统、解释器、包版本和设备，复用兼容环境或在用户批准后准备独立环境。不要把 TensorFlow/PyTorch 旧版本直接安装到工作台 `.venv`。Windows 缺少 Linux 专用依赖时，改在 Linux 或 WSL2 中部署整个执行流程；选择 GPU 前同时检查硬件、驱动、框架与实际计算能力。

## 首次功能检查与常见问题

本地模式直接进入工作台，不注册账号。先创建空项目，再上传论文或说明，核对模型列表并询问上传内容；使用作者源码和可得数据准备方案，确认数据集、命令、环境、设备与预算后再批准运行。代码修改先生成差异预览，应用和新实验分别确认。查看日志、指标、曲线与 Markdown 报告时核对当前运行范围，不将流程完成直接当作全部论文结论复现。

| 情况 | 检查方法与处理 |
|---|---|
| 找不到 Python 或 venv 创建失败 | 确认 3.11/3.12；Linux 安装对应 venv 模块，Windows 安装 Python 启动器或使用绝对解释器路径 |
| 页面 404 或提示缺少前端 | 解压完整运行包；源码部署执行 npm ci、npm run build，确认 frontend/dist/index.html 存在 |
| 端口被占用 | 在 .env 修改 APP_PORT 后重启；生产页面与 API 由同一端口提供。开发模式还需同步 Vite 代理目标 |
| API 返回 401/403/429 | 区分模型接口与工作台账号错误；模型检查授权和额度，网站账号检查会话、同源地址和限流 |
| Conda 不可用 | 在启动终端检查 conda --version 或 CONDA_EXE，不将另一机器的解释器路径照搬到本机 |
| 下载很慢 | 查看传输大小、速度和操作日志；仍有数据传输时继续等待，停滞时按页面原因重试或上传可得文件 |
| 缺少数据 | 补充作者链接、公开数据地址或授权文件；本包不携带第三方数据集，找到同名原始数据仍需核对格式、变体和划分 |
| 空白页面或接口连接失败 | 先看启动终端，再检查前端脚本、后端健康和浏览器控制台；更新完成后重新加载页面 |

只读检查在 `backend` 目录运行 `../.venv/bin/python scripts/check_services.py --frontend http://127.0.0.1:8010 --backend http://127.0.0.1:8010`；Windows 替换为 `..\.venv\Scripts\python`。检查具体项目再加 `--project-id 项目UUID`。这些检查不启动训练，也不能代替浏览器和具体论文的实际验收。

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
