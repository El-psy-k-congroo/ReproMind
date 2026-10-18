# ReproMind

面向开源论文的科研复现工作台，围绕作者源码与上传源码，管理论文、数据、环境、实验与结果，并提供了解项目上下文的复现助手。当前产品不提供独立的“审计”功能。

## 当前能力

- 创建论文项目，上传 PDF、代码 ZIP、配置、结果与说明材料。
- 发现作者仓库，获取 GitHub、Gitee、Zenodo 等来源的代码和数据线索。
- 根据论文、代码及本机平台规划依赖，复用或构建项目独立 Conda 环境。
- 用户确认后运行实验，记录日志、设备、参数、指标与运行曲线，失败时尝试定位和修复。
- 通过复现助手解释论文与代码，预览并确认代码修改、调整实验、运行实验和查看结果。
- 根据可验证的指标及数据划分规划调参，展示曲线和比较图；无法确认的结果保留为未验证。
- 按任务调度多个模型，支持移动云 MoMA 和通用 OpenAI 兼容接口。

作者源码优先。未开源论文自动生成替代实现功能暂缓。模型回答、流程结束或短时运行通过均不等于论文完整复现成功。

## 在线体验与本地部署

[在线评审入口](https://121.196.217.76)（访问口令由维护者单独提供）。

云端评审环境使用独立访问口令，提供项目、材料与复现助手体验。展示服务器为 2核 / 2GB / 无GPU，不执行仓库下载、环境安装、代码修改和实验。工作区由评审人员共享，请勿上传私密材料。**推荐本地部署，体验完整复现流程。**

本地部署无需注册或登录。当前应用不是多租户 SaaS，不应把完整本地模式直接暴露到公网。评审访问口令由维护者单独提供，不能复用服务器 SSH 密码。

## 快速启动

要求 Python **3.11 或 3.12**、Node.js **22.12+**、Git。Conda 仅在创建论文实验环境时需要；启动工作台和使用远程模型接口不要求 GPU、PostgreSQL、Redis、Milvus 或 Docker。

### Windows PowerShell

```powershell
git clone https://github.com/El-psy-k-congroo/ReproMind.git
cd ReproMind
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r backend\requirements.txt
# 仅首次创建配置，不覆盖已有文件。
if (!(Test-Path .env)) { Copy-Item .env.example .env }
cd frontend
npm ci
npm run build
cd ..\backend
..\.venv\Scripts\python run_server.py
```

### Linux

```bash
git clone https://github.com/El-psy-k-congroo/ReproMind.git
cd ReproMind
python3.11 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
test -e .env || cp .env.example .env
cd frontend
npm ci
npm run build
cd ../backend
../.venv/bin/python run_server.py
```

浏览器访问 **http://127.0.0.1:8010**。构建后的前端由后端一起提供，不需要单独启动 Vite。默认数据库为 `backend/data/repromind.db`，路径以启动工作目录 `backend` 为准。

编辑根目录 `.env` 后重启后端。默认使用 SQLite，无需数据库隧道。旧配置若包含不可用的 PostgreSQL 地址，应先备份，再明确选择原数据库或 SQLite，不能直接覆盖旧实验记录。默认 `APP_HOST=127.0.0.1` 仅允许本机访问。

完整实验需要安装 Miniconda / Conda，并按设备准备 CPU 或 GPU 环境。Windows 无法运行的 Linux 专有框架应在 Linux / WSL2 部署；工作台可跨平台运行不代表任意作者项目都兼容两个操作系统。

## 模型配置

在根目录 `.env` 中选择一个提供方并填写自己的 API 密钥。未配置密钥时仍可启动并管理材料，智能问答及模型规划不可用。

```dotenv
MODEL_PROVIDER=moma
MOMA_API_KEY=填写自己的密钥
MOMA_BASE_URL=https://zhenze-huhehaote.cmecloud.cn/v1
MOMA_MODEL_CHAT=填写账号可用模型
MOMA_MODEL_CODER=填写账号可用模型
MOMA_MODEL_REASONING=填写账号可用模型
MOMA_MODEL_ROUTER=填写账号可用模型
```

或者使用通用兼容接口：

```dotenv
MODEL_PROVIDER=openai_compatible
LLM_API_KEY=填写自己的密钥
LLM_BASE_URL=https://你的服务地址/v1
LLM_MODEL_CHAT=填写可用模型
LLM_MODEL_CODER=填写可用模型
LLM_MODEL_REASONING=填写可用模型
```

密钥只放在本地 `.env` 或服务器私有配置，不提交版本库、不进入浏览器。外部接口账号、额度和可用模型由部署者维护。

## 开发与启动检查

开发前端可在 `frontend` 运行 `npm run dev`，访问 5173；API 仍在 8010。默认使用同源 `/api`，Vite 按 `APP_PORT` 或 `VITE_BACKEND_URL` 代理。生产构建不能使用指向开发者 localhost 的绝对 API 地址。

在 `backend` 目录进行只读可用性检查，不启动训练或调用模型：

```bash
python scripts/check_services.py --frontend http://127.0.0.1:8010 --backend http://127.0.0.1:8010
# 检查具体项目：加 --project-id 项目UUID
```

检查分别覆盖前端入口、主脚本、后端健康、前端 API 路径和可选项目接口；还需浏览器实际渲染验收。构建成功不代表服务已启动，后端健康不代表页面可用。

## 轻量云端部署

详见 [部署与运维](docs/deployment.md)。`deploy/` 提供 Nginx、systemd 和证书续期模板。采用 SQLite、单 API 进程和静态前端，不在展示服务器安装深度学习框架或存放训练数据。

本地论文、模型权重、实验记录、上传文件、访问口令、密钥和数据库均不随代码发布。数据库保留的历史表只用于兼容已有记录，不代表存在相应产品功能入口。
