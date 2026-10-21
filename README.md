# ReproMind

ReproMind 是基于多智能体协同的开源论文复现系统。系统把论文、作者代码、数据、环境、命令与结果组织到同一项目中，支持论文理解、实验准备、代码修改、运行纠错、调参与报告生成。

## 本地快速启动

**如果想要运行与 GPU 相关的实验，强烈推荐本地部署。**

比赛运行包已包含构建后的前端。安装 Python 3.11 或 3.12，解压程序包，把 `.env.example` 复制为 `.env` 并填写模型配置，然后：

- Windows：双击 `start-windows.bat`。
- Linux：在解压目录执行 `bash start-linux.sh`。

启动器自动创建应用 `.venv`、安装后端依赖并启动工作台。浏览器打开 **http://127.0.0.1:8010**，本地无需登录。数据库自动创建在 `backend/data/repromind.db`，无需额外部署数据库。

下载作者源码需要 Git，论文独立环境需要 Conda。运行包启动无需 Node.js；从源码构建需要 Node.js 22.12+。

## 从源码启动

```bash
git clone https://github.com/El-psy-k-congroo/ReproMind.git
cd ReproMind
cd frontend
npm ci
npm run build
cd ..
```

在根目录配置 `.env`，使用上述 Windows 或 Linux 启动入口。完整步骤、数据库位置和常见问题见 [本地部署与使用](docs/deployment.md)。

## 模型配置

```dotenv
MODEL_PROVIDER=moma
MOMA_API_KEY=填写自己的密钥
MOMA_BASE_URL=https://zhenze-huhehaote.cmecloud.cn/v1
MOMA_MODEL_ROUTER=填写账号可用模型
MOMA_MODEL_CHAT=填写账号可用模型
MOMA_MODEL_CODER=填写账号可用模型
MOMA_MODEL_REASONING=填写账号可用模型
```

支持移动云 MoMA 和通用兼容接口。模型密钥保存在本地 `.env`，后端负责请求，浏览器只保存能力组的模型选择。

## 主要功能

- 管理论文 PDF、源码 ZIP、配置、数据说明和实验材料。
- 获取作者仓库与数据来源，结合本机平台规划和复用独立环境。
- 复现助手结合论文、源码和运行记录回答问题，发起代码修改和实验需求。
- 展示代码差异，确认后应用修改，并保存关联的新实验。
- 运行作者实验，记录日志、参数、设备和指标；发生异常时定位问题并生成修复。
- 组织调参候选与预算，展示训练曲线、结果比较和复现报告。

用户创建项目后自行上传论文，填写作者仓库和复现目标。网站账号的项目、附件、对话与实验按用户隔离。

## Agent 与技术组成

| 层面 | 技术 | 作用 |
|---|---|---|
| 前端 | Vue 3、TypeScript、Vue Router、Vite | 工作台、项目页面、日志与图表交互 |
| 后端 | Python、FastAPI、Pydantic | 异步接口、参数校验、上传与 SSE |
| Agent 协作 | ReproductionEngine、项目编码服务、Python 异步任务 | 组织论文理解、规划、编码、语义校验、纠错和项目助手的协作 |
| 模型接入 | HTTPX、统一模型路由；LangChain OpenAI SDK 依赖 | 按能力组选择模型；当前主调用链使用 HTTPX 兼容接口 |
| 工作流检查点 | LangGraph、langgraph-checkpoint-postgres | 保留 InMemorySaver 与 PostgreSQL 检查点适配；主流程状态由业务数据库保存 |
| 数据存储 | SQLAlchemy、SQLite、aiosqlite | 本地项目、任务、消息与账号状态，默认启用 WAL |
| 可选数据库 | PostgreSQL、asyncpg、psycopg | 业务数据库连接与检查点存储适配 |
| 实验执行 | Conda、子进程、项目文件系统 | 独立实验环境、代码、数据、终端日志和报告 |

Agent 的输入、输出与交接关系见 [多 Agent 协作设计](docs/multi-agent-architecture.md)。依赖清单见 `backend/requirements.txt` 和 `frontend/package-lock.json`，前端版权与许可文本见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
