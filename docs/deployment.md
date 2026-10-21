# ReproMind 本地部署与使用

ReproMind 支持 Windows 和 Linux。本地使用无需注册登录，数据库随应用自动创建。

**如果想要运行与 GPU 相关的实验，强烈推荐本地部署。** 请在具备相应显卡、驱动和 CUDA 条件的机器上运行作者实验。模型 API 负责推理，论文训练在部署机器上执行。

## 一、准备软件

| 用途 | 需要的软件 |
|---|---|
| 启动比赛运行包 | Python 3.11 或 3.12，安装时启用 pip 和 venv；首次启动需要联网 |
| 下载作者仓库 | Git，并确保终端可以执行 `git --version` |
| 创建论文实验环境 | Miniconda、Miniforge 或 Conda，并确保 `conda --version` 可用 |
| 从源码构建前端 | Node.js 22.12+；比赛运行包已经构建前端，无需安装 Node.js |

Windows 安装 Python 时选择 Add Python to PATH。Linux 如果提示缺少 venv，请安装对应解释器的 venv 软件包，例如 Ubuntu 的 `python3.11-venv`。

## 二、运行包一键启动

1. 解压 `ReproMind_跨平台运行包.zip`，进入其中的 `ReproMind` 文件夹。
2. 将 `.env.example` 复制为 `.env`。把 MoMA 的接口密钥和模型配置填入 `.env`；比赛材料中的《评审专用MoMA配置》提供可直接复制的配置。
3. Windows 双击 **start-windows.bat**；Linux 在该文件夹执行 **`bash start-linux.sh`**。
4. 第一次启动自动创建应用 `.venv`、安装后端依赖并初始化 SQLite 数据库。终端显示服务监听后，打开 **http://127.0.0.1:8010**。
5. 后续直接使用同一个启动入口。保留终端窗口，停止工作台时按 Ctrl+C。

Windows 也可以在 PowerShell 执行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\start-windows.ps1`。Linux 可以用 `python3.11 start.py` 或 `python3.12 start.py` 指定解释器。启动器保留已有 `.env` 和数据。

## 三、从源代码启动

先安装上表中的 Python、Git 和 Node.js，然后执行：

```bash
git clone https://github.com/El-psy-k-congroo/ReproMind.git
cd ReproMind
cd frontend
npm ci
npm run build
cd ..
```

在根目录按第二节配置 `.env`，Windows 双击 `start-windows.bat`，Linux 执行 `bash start-linux.sh`。

## 四、MoMA 配置

```dotenv
MODEL_PROVIDER=moma
MOMA_API_KEY=填写接口密钥
MOMA_BASE_URL=https://zhenze-huhehaote.cmecloud.cn/v1
MOMA_MODEL_ROUTER=填写账号可用模型
MOMA_MODEL_CHAT=填写账号可用模型
MOMA_MODEL_CODER=填写账号可用模型
MOMA_MODEL_REASONING=填写账号可用模型
```

后端按任务角色选择模型，通过 HTTPS 接口调用。密钥保存在本地配置，不传给前端。配置修改后重启工作台。

## 五、本地数据库与文件

默认数据库为 **backend/data/repromind.db**，通过 SQLAlchemy 和 aiosqlite 读写，启用 SQLite WAL。应用第一次启动时自动建立目录和数据表，无需安装 PostgreSQL 或连接网站数据库。

上传目录和项目目录由 `.env` 中的 `UPLOAD_DIR`、`PAPER_PROJECTS_DIR` 决定，默认保存在本机。每台机器的项目独立保存。启动器固定在 backend 目录运行，因此默认相对路径保持一致。备份时先停止活动实验和工作台，再备份数据库、附件、项目目录及 `.env`。

## 六、第一次体验

1. 首页点击“新建复现项目”，填写项目名称，进入项目后上传自己的论文 PDF，填写作者仓库和复现目标。
2. 在复现助手中询问论文方法、代码入口、数据划分或超参数。
3. 点击启动自主复现，核对作者仓库、所选数据集、设备和环境。建议先选择 CPU 上的 Cora 实验。
4. 准备完成后查看执行计划，确认后运行；通过流水线、终端日志、曲线和结果卡片观察过程。
5. 提出修改需求时说明数据集、目标参数和设备，查看代码差异后确认应用，再运行修改实验。
6. 在运行记录中下载复现报告，比较实际参数、论文目标和结果。

## 七、常见问题

| 情况 | 处理方式 |
|---|---|
| 找不到 Python | 安装 3.11 或 3.12，重新打开终端；Windows 检查 `py -3.11 --version` |
| 依赖安装中断 | 检查网络与包源，重新运行启动入口，已安装的依赖会复用 |
| 缺少 frontend/dist | 使用完整运行包，或按第三节执行前端构建 |
| 8010 端口占用 | 在 `.env` 设置其他 `APP_PORT`，重启后使用新端口访问 |
| 模型请求失败 | 检查 MoMA 密钥、模型名称、服务额度和网络 |
| 找不到 Conda | 在同一终端检查 `conda --version`；必要时设置 `CONDA_EXE` 为本机 Conda 路径 |
| 作者项目仅支持 Linux | 使用 Linux 或配置好相应运行条件的 WSL2，再按作者入口准备实验环境 |
| 数据下载较慢或缺文件 | 查看下载日志，补充作者公开链接或上传相应文件，核对数据格式与划分 |

论文环境由工作台结合作者依赖、本机系统和设备准备，不要复制其他机器的 `.venv` 或 Conda 环境。
