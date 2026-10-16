# ReproMind

面向科研复现与实验可信审计的多智能体协作平台。系统把论文、代码、配置和实验结果放在同一条证据链上，自动生成复现计划、发现配置与指标异常，并输出可追溯的审计报告。

## 当前能力

- 项目级科研工作台与四类证据上传
- PDF、Markdown、文本、YAML、JSON、CSV 与 ZIP 文件解析
- LangGraph 驱动的六节点多 Agent 工作流
- Research Orchestrator 根据任务与证据执行智能路由
- Paper、Code、Reproduction、Audit、Report 专业 Agent
- 按 Agent 角色调度 router/chat/coder/reasoning 模型
- 移动云 MoMA 与通用 OpenAI 兼容网关双提供方
- 论文/配置/结果的确定性规则审计与模型交叉核验
- 风险分级、复现评分、证据定位和建议动作
- PostgreSQL 业务持久化与 LangGraph checkpoint；SQLite 仅作离线降级
- 论文仓库链接自动发现、GitHub/Gitee 源码获取与上传 ZIP 解包
- 当前优先支持作者开源仓库与上传源码；未开源论文自动编码已暂缓，不自动生成替代实现
- 每篇论文一个持久项目工作区，不同仓库与生成实现使用独立子目录
- 按模型名创建项目内 Conda 环境；依赖文件优先，无依赖文件时分析 import 推断
- 人工批准后的环境构建、受限命令执行、日志诊断、依赖修复与自动重试

## 目录

```text
backend/app/agents/        独立 Agent、Prompt、节点与共享 State
backend/app/orchestrator/  LangGraph 状态图与编排服务
backend/app/core/          多模型路由和上下文管理
backend/app/services/      文件解析与确定性审计工具
backend/scripts/           数据库初始化与连通性检查脚本
frontend/                  Vue 3、TypeScript、Vite
```

## 多 Agent 架构

```text
Research Orchestrator (router model)
        │
        ├── paper_review ─────────── Paper ───────────── Report
        ├── reproduction_plan ────── Paper → Code → Reproduction → Report
        ├── result_audit ─────────── Audit ───────────── Report
        └── full_audit ───────────── Paper → Code → Reproduction → Audit → Report

模型角色：
Paper / Report  → chat model
Code / Reproduction → coder model
Audit → reasoning model
Orchestrator → router model
```

所有 Agent 共享显式 `ResearchState`。上下文管理器负责证据清单、按类型分组、
内容长度控制和 SHA-256 版本标识；LangGraph Checkpointer 使用项目与分析记录组成的
`thread_id` 隔离运行上下文。每次模型选择、耗时、Token 用量、降级情况都会写入报告轨迹。

## 自主复现流水线

```text
论文链接识别 → 作者源码获取 → 论文实验信息与代码核验 → 环境规划
        → 数据准备 → 人工确认 → 实验执行 → 结果核验 → 报告
```

系统不使用 Docker。每篇论文使用一个持久的 `data/paper-projects/<项目名>-<项目ID>/`
工作区，其中包含 `paper/`、`sources/`、`generated/`、`data/`、`.conda/`、`runs/` 和
`reports/`。不同代码来源进入不同子目录，多次运行复用论文工作区和模型环境。

外部命令不通过 shell 拼接，执行程序受允许列表和超时限制。未知论文代码在用户明确批准前不会运行；
批准后，Environment Agent 按模型名在项目内建立独立 Conda prefix，不修改 `eduagent` 环境。
Hugging Face 数据地址统一配置为 `https://hf-mirror.com`。Git 拉取会对临时网络错误重试；
遇到 Windows 不允许的仓库路径时，从 Git 对象库导出可运行源码。源码获取失败时保留实际诊断，
允许重试、提供正确作者仓库或上传源码，不自动转入代码生成。作者源码流程不受论文驱动编码的
方法范围短引文校验拦截；数据集、指标、数据可用性和执行命令仍需核验，不能伪造实验目标。
实验批准前只准备源码、数据线索和环境计划，不安装依赖、不启动训练。后续实验复用并检查项目环境。

## 快速启动

### 1. 建立服务器隧道

```powershell
ssh -L 15432:127.0.0.1:5432 -L 29530:127.0.0.1:19530 pjx@10.10.80.252
```

当前业务只要求 PostgreSQL；Milvus 配置已预留，Redis、MinIO 暂不强制启用。

### 2. 后端

```powershell
conda activate eduagent
cd backend
pip install -r requirements.txt
Copy-Item ..\.env.example ..\.env
# 在本机 .env 中填写数据库密码与 MoMA token，不要提交该文件
python run_server.py
```

首次创建独立数据库时，在 `backend` 目录执行：

```powershell
$env:ADMIN_DATABASE_URL='postgresql://USER:PASSWORD@127.0.0.1:15432/eduagent'
python scripts/bootstrap_postgres.py
```

### 3. 前端

```powershell
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173`，创建论文复现项目并上传论文即可开始。

前端与后端是两个独立服务；`npm run build` 只生成构建文件，不会启动前端。页面显示
`ERR_CONNECTION_REFUSED` 时，应先检查 5173 是否有前端进程监听，不能仅检查后端健康。
可在 `backend` 目录运行只读连通性验收（不会请求模型或启动实验）：

```powershell
python scripts/check_services.py
# 同时验证当前项目：
python scripts/check_services.py --project-id 项目UUID
```

脚本分别检查前端入口、主脚本、后端健康、前端 API 代理及可选项目接口；任一失败返回非零退出码。
这些 HTTP 检查不能替代浏览器实际渲染验收。

数据下载默认总时限为 30 分钟，连续 3 分钟收不到网络数据才判定读取超时，
连接等待为 60 秒。在根目录 `.env` 中可分别设置
`DATASET_DOWNLOAD_TIMEOUT_SECONDS`、`DATASET_DOWNLOAD_READ_TIMEOUT_SECONDS` 和
`DATASET_DOWNLOAD_CONNECT_TIMEOUT_SECONDS`（单位为秒，修改后重启后端生效）。
总时限对单次文件下载及其来源重试生效；内置 torchvision 下载按下载进程计算。
这些设置不改变训练时限、执行批准和下载文件安全检查。

## 模型配置

在根目录 `.env` 中填写：

```dotenv
MODEL_PROVIDER=moma
MOMA_API_KEY=your-moma-key
MOMA_BASE_URL=https://zhenze-huhehaote.cmecloud.cn/v1
MOMA_MODEL_ROUTER=deepseek-v4-flash-0731
MOMA_MODEL_CHAT=deepseek-v4-flash-0731
MOMA_MODEL_CODER=deepseek-v4-flash-0731
MOMA_MODEL_REASONING=deepseek-v4-flash-0731
```

密钥不会进入前端，也不应提交到版本库。模型调用失败时，规则审计仍可运行。

通用 OpenAI 兼容网关仍可用于本地开发：

```dotenv
MODEL_PROVIDER=openai_compatible
LLM_API_KEY=your-key
LLM_BASE_URL=https://api.chatanywhere.tech/v1
LLM_MODEL_CHAT=gpt-4o-mini
LLM_MODEL_CODER=gpt-4o-mini
LLM_MODEL_REASONING=gpt-4o-mini
```

模型路由按 router/chat/coder/reasoning 逻辑角色设计。当前可以让四个角色共用一个低成本模型，后续在 MoMA 获得更多模型后只需修改环境变量，无需改 Agent 代码。单次输出默认限制为 1600 token，证据上下文也有总量上限。

## 数据库

详细表关系、字段职责和基础设施取舍见 [数据库设计](docs/database-design.md)。ReproMind 使用独立 `repromind` 数据库，不修改 EduAgent 现有表。

## 生产部署建议

- 将 `DATABASE_URL` 改为 PostgreSQL 连接串。
- 将本地上传目录替换为对象存储适配器。
- 在反向代理中限制上传大小并启用 HTTPS。
- 生产部署时应将执行 Worker 与 API 进程隔离；当前无 Docker 版本通过独立目录、命令允许列表、超时和人工确认降低风险，但不等同于操作系统级沙箱。
