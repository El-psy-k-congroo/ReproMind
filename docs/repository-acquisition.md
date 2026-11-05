# 作者源码获取与恢复

## 获取顺序

1. 复用当前项目中已完成获取并记录仓库身份的源码，保留代码修改及历史目录。只有来源地址、没有完成标记的残缺 Git 目录不能作为成功缓存。
2. GitHub 仓库优先下载官方默认分支源码包，不再依次猜测 `master`、`main`。首选 codeload 的 `HEAD`，备用 REST 接口省略分支参数。
3. 下载通道出现超时、中断、限流或损坏压缩包时，进行有次数及总时间限制的重试。默认先遵循系统代理配置，再尝试同一官方来源的直连；不会改动全局代理或关闭 TLS 验证。配置 `REPRODUCTION_USE_SYSTEM_PROXY=false` 时始终直连，同一地址不会重复尝试相同的直连方式。
4. 压缩包通道未完成时回退到浅克隆，最多两次。首次沿用系统代理配置，第二次明确直连并使用 HTTP/1.1；进程内覆盖代理配置，不改用户全局 Git 设置。只有 Windows 路径导致的检出失败才尝试从 Git 对象恢复，不从传输超时的空对象库执行 `git archive HEAD`。

源码包下载的总预算为 `GIT_CLONE_TIMEOUT_SECONDS` 的两倍，范围为 30—120 秒。每轮将剩余预算按仍可尝试的通道分配，避免一个持续缓慢返回少量数据的通道耗尽全部预算；未使用的时间留给后续通道。单次连接最多 8 秒，无数据读取最多 15 秒。Git 克隆每次仍使用原有超时配置，并限制持续低速无效传输。每个阶段都可响应暂停，不会在暂停后切换到另一个下载通道。

## 完整性与失败恢复

- 使用流式写入，下载中即检查 120 MB 体积上限，并核对声明长度、ZIP 结构、CRC、文件数量和解压体积。完成后才发布归档及源码缓存标记。
- 缓存标记记录仓库地址、获取通道、归档 SHA256、体积，以及官方地址可确定的提交编号；无法确定提交编号时留空。
- 失败的下载临时文件会清理；原源码、旧归档和失败诊断保留。重试使用独立路径，成功生成的新路径同样可供下一次复用。
- 源码获取失败时停在等待输入状态，显示明确的重试、修改仓库地址或上传作者代码入口。原始诊断折叠显示，重试前保存到准备历史。
- 重试源码获取只重新准备实验，不批准安装环境或执行训练，也不生成替代作者实现。

## 官方行为核对

排查使用 `web-debug-search` 的通用网页来源，仅读取 GitHub 与 Git 官方文档；社区、问答来源未使用。资料说明接口及参数行为，网络是否可用以本机下载验证为准。

| 匹配程度 | 类型 | 用途 | 权威性 | 来源 | 适用范围 | 结论 | 状态 |
|---|---|---|---|---|---|---|---|
| `[CONTEXTUAL]` | `[API-USAGE]` | `[DEBUGGING-ONLY]` | `[OFFICIAL]` | [GitHub 仓库归档接口](https://docs.github.com/en/rest/repos/contents#download-a-repository-archive-zip) | GitHub 公共仓库 REST API | 省略 ref 使用默认分支，客户端需要跟随重定向。 | 已阅读；不是截图故障的直接复现记录 |
| `[CONTEXTUAL]` | `[API-USAGE]` | `[DEBUGGING-ONLY]` | `[OFFICIAL]` | [Git clone 文档](https://git-scm.com/docs/git-clone) | Git 命令行 | 浅克隆、单分支和无标签选项可减少传输；正常克隆使用远端 HEAD 指向的分支。 | 已阅读；不能保证任意网络或仓库均可访问 |
| `[CONTEXTUAL]` | `[API-USAGE]` | `[DEBUGGING-ONLY]` | `[OFFICIAL]` | [HTTPX 超时](https://www.python-httpx.org/advanced/timeouts/) | Python HTTPX | 读取超时约束等待下一块数据的时间，不能替代整个通道的时间预算。 | 已核对；慢速首通道阻塞回退另有离线回归 |
| `[CONTEXTUAL]` | `[API-USAGE]` | `[DEBUGGING-ONLY]` | `[OFFICIAL]` | [HTTPX 环境变量](https://www.python-httpx.org/environment_variables/) | Python HTTPX | 代理来自运行环境，`trust_env=False` 忽略环境配置。 | 已核对；本机网络质量以实际请求为准 |
| `[CONTEXTUAL]` | `[API-USAGE]` | `[DEBUGGING-ONLY]` | `[OFFICIAL]` | [Git 进程配置](https://git-scm.com/docs/git-config) | Git HTTP 传输 | `GIT_CONFIG_COUNT/KEY/VALUE` 可为当前进程覆盖代理配置，避免写入全局配置及把代理凭据放进参数。 | 已核对；代理与直连回退有离线回归 |

> **Evidence boundary:** These GitHub, Q&A, community, and general-web results are for debugging and discovery only. They are not paper-citation evidence and must not be added to the bibliography or used alone to support a research claim. Use the project's literature and citation-verification workflow for that purpose.
