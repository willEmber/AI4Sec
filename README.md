<p align="center">
  <img src="scholar.png" alt="Scholar logo" width="160" />
</p>

<h1 align="center">Scholar Platform · 学术论文精读与研究平台</h1>

<p align="center">
  从一个研究问题出发，或直接上传多篇 PDF——在同一个多轮对话里完成检索、获取、解析、精读与对比。<br/>
  每一条结论都绑定<strong>可回溯原文证据</strong>，一键跳回 PDF 对应页码核验，让 AI 阅读真正"有据可查"。
</p>

<p align="center">
    <a href="https://linux.do/t/topic/2108966/20" alt="LINUX DO">
        <img src="https://img.shields.io/badge/LINUX-DO-FFB003.svg?logo=data:image/svg%2bxml;base64,DQo8c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAiIHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIj48cGF0aCBkPSJNNjguMi0uMDU1aDYuMjVxMjMuOTY5IDIuMDYyIDM4IDIxLjQyNmM1LjI1OCA3LjY3NiA4LjIxNSAxNi4xNTYgOC44NzUgMjUuNDV2Ni4yNXEtMi4wNjQtMjMuOTY4LTIxLjQzIDM4LTExLjUxMiA3Ljg4NS0yNS40NDUgOC44NzRoLTYuMjVxLTIzLjk3LTIuMDY0LTM4LjAwNC0yMS40M1EuOTcxIDY3LjA1Ni0uMDU0IDUzLjE4di02LjQ3M0MxLjM2MiAzMC43ODEgOC41MDMgMTguMTQ4IDIxLjM3IDguODE3IDI5LjA0NyAzLjU2MiAzNy41MjcuNjA0IDQ2LjgyMS0uMDU2IiBzdHlsZT0ic3Ryb2tlOm5vbmU7ZmlsbC1ydWxlOmV2ZW5vZGQ7ZmlsbDojZWNlY2VjO2ZpbGwtb3BhY2l0eToxIi8+PHBhdGggZD0iTTQ3LjI2NiAyLjk1N3EyMi41My0uNjUgMzcuNzc3IDE1LjczOGE0OS43IDQ5LjcgMCAwIDEgNi44NjcgMTAuMTU3cS00MS45NjQuMjIyLTgzLjkzIDAgOS43NS0xOC42MTYgMzAuMDI0LTI0LjM4N2E2MSA2MSAwIDAgMSA5LjI2Mi0xLjUwOCIgc3R5bGU9InN0cm9rZTpub25lO2ZpbGwtcnVsZTpldmVub2RkO2ZpbGw6IzE5MTkxOTtmaWxsLW9wYWNpdHk6MSIvPjxwYXRoIGQ9Ik03Ljk4IDcwLjkyNmMyNy45NzctLjAzNSA1NS45NTQgMCA4My45My4xMTNRODMuNDI2IDg3LjQ3MyA2Ni4xMyA5NC4wODZxLTE4LjgxIDYuNTQ0LTM2LjgzMi0xLjg5OC0xNC4yMDMtNy4wOS0yMS4zMTctMjEuMjYyIiBzdHlsZT0ic3Ryb2tlOm5vbmU7ZmlsbC1ydWxlOmV2ZW5vZGQ7ZmlsbDojZjlhZjAwO2ZpbGwtb3BhY2l0eToxIi8+PC9zdmc+" alt="LINUX DO" /></a>
    <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/LangGraph-1C3C3C?logo=langchain&logoColor=white" alt="LangGraph" />
    <img src="https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL" />
    <img src="https://img.shields.io/badge/Next.js-15-000000?logo=nextdotjs&logoColor=white" alt="Next.js" />
    <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black" alt="React" />
    <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker" />
</p>

<p align="center">
  <a href="./README.en.md">English README</a>
</p>

## 预览

<p align="center">
  <img src="example.png" alt="Scholar 预览" width="760" />
</p>

## 核心特性

- 💬 **统一对话阅读智能体（`/chat`）**：基于 LangGraph + Deep Agents 的单主循环智能体，自主调用 **7 大类 22 个领域工具**完成找论文、下全文、解析、章节/公式/表格精读、多文对比与生成结构化报告；内置持久检查点、Worker 崩溃恢复、预算触顶收尾与断线事件续传。
- 🔍 **不可变证据与段落级溯源**：全文与外部信息均记录为绑定解析版本的不可变证据快照；模型在会话内通过短别名（`[E12]`）精准引用并由服务端实时还原，点击引用徽章即可预览原文摘录、切换对应论文并跳至准确页码，从机制上拦截幻觉与假引用。
- 🎯 **三大整篇深度报告 + 横向对比矩阵**：内置 **Insight Snap**（结合外部引用/撤稿/开源信号的快速初筛）、**Logic Lens**（公式、算法、实验复现拆解与定向焦点分析）、**Research Sphere**（参考文献网络、引用图谱与研究空白）。在对话中作为工具按需调用（输入框里一键生成）；每份报告有独立的报告页与导出，多篇 Snap 可在 `/library` 的对比视图中横向比较。
- 🗂️ **研究项目与跨会话回忆（`/projects`）**：按课题归集多轮对话、项目论文与长期偏好记忆（含待确认防注入保护）；支持中英文全文检索历史对话，将过往结论连同当时的原始支撑证据一并召回复用。
- 🌐 **7 源学术检索 + 通用网页查证**：并行检索 Semantic Scholar、OpenAlex、arXiv、Crossref、IEEE Xplore、DBLP、OpenReview 及期刊分区（EasyScholar + Web 兜底，严格区分 JCR / 中科院 / CCF）；集成 Tavily / Exa / Firecrawl 密钥池查证开源代码库、排行榜与项目主页。
- 📄 **高精度版面解析**：基于 MinerU 解析 PDF，完整保留章节层级、公式、宽表与配图；附加 PDF 即后台预解析，与打字提问并行不阻塞。
- 📚 **自托管知识库 RAG（`/library`）**：对接 Dify 知识库检索代理，在私有论文语料库上进行全文 / 语义 / 混合检索与跨文献综合问答。
- 📤 **一键离线导出**：导出 **Zotero 导入包（`.zip`）**（自动补全元数据 + 原生 KaTeX 公式报告子笔记 + PDF 附件）或便携 **Markdown**；可选经 R2 / S3 对象存储将报告配图托管为公共链接（详见[导出](#导出export)）。
- 🔐 **多模型网关、账号配额与链路追踪**：支持百炼基座 + New-API 等扩展对话网关（[`models.example.toml`](./backend/models.example.toml)）；提供单用户免登录模式或 GitHub / Google / LINUX DO OAuth 多用户模式、按日运行/Token 配额与数据保留期清理；可选开启 Arize Phoenix 追踪完整 Prompt 与工具调用链。

## 功能入口与分析模式

| 入口 / 模式 | 适用场景 |
|---|---|
| **阅读对话（`/chat`）** | 平台主入口：无附件从零检索文献并下载精读，或围绕一篇/多篇 PDF 连续追问、跨文对比、查证代码与同行评审；下方三种整篇报告由对话按各自的方法阅读后写成，每条结论引用原文段落 |
| **Insight Snap（快速洞察）** | 约两分钟的证据化初筛：核心贡献拆解 + 独立盲审 + 外部客观信号（发表场所等级、S2/OpenAlex 引用与意图、OA/撤稿状态、代码仓库）→ 综合阅读建议；支持在 `/library` 的对比视图横向对比多篇论文 |
| **Logic Lens（逻辑透镜）** | 深度方法与复现拆解：问题定义、方法管线、排版级核心公式卡片、算法步骤、实验数据表与复现检查清单；支持传入具体关注点（`focus`）定向深挖，提供结构化卡片与全文双视图 |
| **Research Sphere（研究全景）** | 领域脉络定位：抽取参考文献 + 引用图谱扩展 → 期刊/被引评分与 LLM 相关性门控 → 耦合聚类得到文献地图（含私有库相关工作匹配），对话据此写出主题、多论文对比与研究空白 |
| **研究项目（`/projects`）** | 课题级工作区：管理同一研究方向下的多轮会话、共享论文池、课题专属记忆与历史问答检索 |
| **知识库（`/library`）** | 私有语料库检索与跨文献问答（配置 `DIFY_API_BASE` 后启用） |

## 知识库（Knowledge Base）

在前端「知识库」页面，可在你自建的论文语料库上进行：

- **检索（Search）**：在整个语料库内全文 / 语义 / 混合检索，支持文档预览与原文跳转。
- **问答（Ask）**：跨多篇论文提问，由 LLM 基于检索片段综合作答并给出来源。

该功能由自托管的 Dify 知识库检索代理提供支撑。设置 `DIFY_API_BASE` 即可启用；留空则自动关闭「知识库」页面与 Research Sphere 中的「库内相关工作」匹配，其余功能不受影响。

## 导出（Export）

在报告结果页或对话报告卡片中可将精读成果一键导出，便于归档与文献管理：

- **Zotero 导入包（`.zip`）**：在本地 Zotero 中 `文件 → 导入` 该 `.zip`，即可得到一个 `journalArticle` 条目、一条 AI 报告子笔记、以及作为附件存储的 PDF 原文——全程离线，无需 Zotero 云账号。
  - **元数据自动补全**：作者、摘要、卷期页码、ISSN、DOI 等通过多源（Crossref → OpenAlex，按 DOI 或标题）尽力补齐，避免条目只有标题/期刊/日期。
  - **笔记标题清晰**：笔记首行即 Zotero 中显示的笔记名，使用「模式 + 论文标题」（问答模式则用你的问题）。
  - **公式正常渲染**：报告中的 LaTeX 转为 Zotero 笔记编辑器原生的数学节点（行内 `<span class="math">`、块级 `<pre class="math">`），由 Zotero 用 KaTeX 渲染，不再显示为裸 `$...$`。
- **便携 Markdown（`.md`）**：导出可在 App 外正常查看的 Markdown。

**图片托管（可选）**：报告中的图片原本是 App 内部链接，离开应用后无法显示。配置对象存储（R2 / S3 兼容）后，导出时图片会自动上传并在 Zotero 笔记与 Markdown 中替换为公共链接；未配置时，Zotero 笔记会将图片以 base64 内嵌（离线兜底），Markdown 则保留原链接。配置见下方 `R2_*` 变量。

## 快速开始（Docker）

### 前置要求

- Docker >= 24.0
- Docker Compose >= 2.20

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入必填 API Key 和模型名称
```

**必填变量**（核心模型与 PDF 解析）：

| 变量 | 说明 |
|---|---|
| `LLM_BASEURL` | 基座网关地址（需支持 OpenAI Responses API `POST {base}/responses`，如百炼 MaaS 兼容端点） |
| `LLM_APIKEY` | 基座网关 API Key |
| `THINKING_MODELNAME` | 可选对话/报告推理模型列表（逗号分隔，第一个为默认，前端提供下拉切换） |
| `MINERU_TOKEN` | MinerU PDF 解析 API Token |

> Docker Compose 会自动拉起内置的 `pgvector/pgvector:pg17` 数据库并拼装连接串，无需手动填写 `DATABASE_URL`（仅建议在生产环境中修改 `POSTGRES_PASSWORD`）。

**常用可选变量**（按需启用对应能力）：

| 变量 | 启用的功能 |
|---|---|
| `BASE_MODELNAME` / `RERANK_MODELNAME` | 内部轻量辅助任务模型（意图分类、摘要抽取、上下文压缩等，默认关闭思考）与检索重排序模型 |
| `TAVILY_API_KEYS` / `EXA_API_KEYS` / `FIRECRAWL_API_KEYS` | 智能体通用网页检索、网页正文抓取与期刊分区兜底查询（支持逗号分隔的多 Key 轮换池） |
| `KEENABLE_API_KEYS` | 可选，仅用于网页检索：排在检索链最后，前面的服务都未完成检索时才调用（同样支持逗号分隔的多 Key 轮换池） |
| `EASYSCHOLAR_SECRET_KEY` | 期刊/会议分级查询（SCI / CCF / CSCD 分区） |
| `S2_API_KEY` / `OPENALEX_KEY` / `PAPERSEARCH_IEEE_API_KEY` | 提升学术检索配额与速率上限，启用 IEEE Xplore 检索；配置 `OPENREVIEW_USERNAME` / `OPENREVIEW_PASSWORD` 可读取同行评审全文 |
| `UNPAYWALL_EMAIL` / `CORE_API_KEY` / `ELSEVIER_API_KEY` / `ELSEVIER_INSTTOKEN` / `WILEY_TDM_TOKEN` | 扩展论文全文自动获取渠道（未配置时仍支持 arXiv 与检索源自带 OA 链接下载） |
| `AUTH_MODE` | `multi_user`（默认，支持匿名试用 + OAuth 登录 + 每日配额）或 `single_user`（个人私有部署，免登录无配额限制） |
| `OAUTH_{GITHUB,GOOGLE,LINUXDO}_CLIENT_ID` / `_CLIENT_SECRET` | 启用对应第三方 OAuth 登录，登录时自动合并当前浏览器的匿名会话与数据 |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` / `ADMIN_API_TOKEN` | 用户名密码登录的管理员账号（免配额、可访问 `/api/admin/*` 与 Phoenix 追踪面板）及管理接口 Token |
| `DIFY_API_BASE` | 启用「知识库」页面与 Research Sphere 库内相关工作匹配 |
| `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_BUCKET` / `R2_ENDPOINT` / `R2_PUBLIC_BASE_URL` | 导出时将报告配图上传至 R2 / S3 兼容对象存储并替换为公共链接 |
| `TRACING_OTLP_ENDPOINT` / `TRACING_UI_URL` | 配合 `--profile tracing` 将 LangChain / LangGraph 调用链导出至自托管 Phoenix，并在 `<前端地址>/api/admin/phoenix` 对管理员开放 |
| `ENABLE_DOCS` | 生产环境可设为 `false` 以关闭 Swagger / OpenAPI 文档 |

**接入扩展模型网关（可选）**：若需接入自建 New-API 等 `/chat/completions` 网关下的其他对话模型（如 Gemini、Claude 等），可将 [`backend/models.example.toml`](./backend/models.example.toml) 复制为 `backend/models.toml`，在 `.env` 中配置对应网关地址与密钥，并通过 `verify_model_gateway` 验证后启用。完整配置项及默认值说明见 [`.env.example`](./.env.example)。

### 2. 构建并启动

```bash
# 标准启动（PostgreSQL + 后端 + 前端）
docker compose up -d

# 可选：同时启动 Arize Phoenix 链路追踪面板（仅绑定本机 127.0.0.1:6006，经管理员鉴权访问）
docker compose --profile tracing up -d
```

- 前端入口：http://localhost:3001（默认进入 `/chat` 对话工作台）
- 后端 API：http://localhost:8001
- API 文档：http://localhost:8001/docs

### 3. 停止服务

```bash
docker compose down
```

### 数据持久化

- **数据库与对话检查点**：存储在 Docker 命名卷 `scholar-pgdata` 中（避免宿主机目录权限变更导致 PostgreSQL 拒绝启动）。
- **PDF 原文与 MinerU 解析产物**：持久化到宿主机的 `./docker-data/` 目录（首次启动自动创建）。
- **Phoenix 追踪数据**（启用 `tracing` profile 时）：存储在命名卷 `scholar-phoenix` 中。

### 自定义基础镜像 / 镜像加速

可通过 `.env` 中的变量替换三个服务的基础镜像（例如在中国大陆环境下指向镜像加速源）：

```ini
POSTGRES_IMAGE=pgvector/pgvector:pg17
PYTHON_IMAGE=python:3.13-slim
NODE_IMAGE=node:22-alpine
```

### 代码或构建期配置变更后重新构建

```bash
# 重新构建并启动全部服务
docker compose up -d --build

# 仅重新构建后端（例如修改了后端代码或 backend/models.toml）
docker compose up -d --build backend

# 仅重新构建前端（例如修改了前端代码或 NEXT_PUBLIC_BACKEND_URL）
docker compose up -d --build frontend
```

## 本地开发

本地裸机开发需要一个可连接的 PostgreSQL 16+（安装 `pgvector` 扩展），并在根目录 `.env` 中配置 `DATABASE_URL`（如 `postgresql://scholar:scholar@localhost:5432/scholar`）。

### 一键启动（推荐）

```bash
./scripts/dev.sh
```

同时拉起后端（`:8000`，`--reload` 热重载）与前端（`:3000`，HMR 热更新），Ctrl-C 一起退出。脚本会自动同步依赖、将前端 `/api/*` 代理与 SSE 地址指向本次后端端口，并将日志写入 `.dev-logs/`。

```bash
./scripts/dev.sh --backend            # 只起后端
./scripts/dev.sh --frontend           # 只起前端（连已在运行的后端）
./scripts/dev.sh --kill-port          # 端口被占用时自动结束占用进程
./scripts/dev.sh --backend-port 8010 --frontend-port 3010
./scripts/dev.sh --help
```

### 手动启动

```bash
# 后端（FastAPI）
cd backend
uv sync
CORS_ORIGINS='["http://localhost:3000"]' uv run uvicorn app.main:app --reload --port 8000

# 快速导入自检
uv run python -c "from app.main import app; print('OK')"
```

```bash
# 前端（Next.js）—— 两个变量都要指向后端端口，否则默认请求 :8001（Docker 端口）：
#   BACKEND_URL             服务端，用于 /api/* 同源代理
#   NEXT_PUBLIC_BACKEND_URL 浏览器端，用于 EventSource 直连 SSE 事件流（避免代理缓冲）
cd frontend
npm install
BACKEND_URL=http://localhost:8000 \
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 \
npm run dev
```

### 测试与离线评测

```bash
# 后端单元测试（设置 TEST_DATABASE_URL 时包含全部 PostgreSQL 隔离 schema 测试；未设置时仅运行纯单元测试）
cd backend
TEST_DATABASE_URL=postgresql://postgres@localhost:5433/scholar_test \
  uv run python -m unittest discover -s tests -p 'test_*.py'

# 离线校验阅读与调用链评测集（不消耗模型 Token）
uv run python -m evals.run --validate
uv run python -m evals.run --dataset chains_seed --validate

# 前端类型检查、单元测试与生产构建校验
cd ../frontend
npx tsc --noEmit
node --test tests/*.test.cjs
npm run build
```

## 架构概览

```
对话主路径（/chat）
  用户提问 / 上传 PDF / 选择项目
      → FastAPI 会话与配额鉴权 → PostgreSQL 持久队列 & Worker 心跳租约
      → Deep Agents 单主循环（TurnGuard 预算/超时兜底 · 上下文摘要压缩 · 旧工具结果淘汰）
          ├─ 发现与查证：7 源学术检索 · 元数据/评审/相关文献 · EasyScholar 期刊分级 · Web 搜索与抓取
          ├─ 获取与解析：arXiv / OA / TDM 下载 → MinerU 解析 → PaperIR（章节树 + 块 + 检索索引）
          ├─ 全文精读：章节大纲 · 跨文检索 · 整节/宽表/公式读取 → 不可变 Evidence 快照（[E12] ↔ [ev_...]）
          ├─ 整篇报告工具：Insight Snap / Logic Lens（支持 focus）/ Research Sphere
          └─ 积累与复用：课题项目论文池 · 跨会话回忆（带证据复用）· 长期偏好记忆
      → 事件落库后经 SSE 实时推送（支持断线按序号续传）
      → 三栏工作台：会话导航（左）+ 回合时间线与可点击引用（中）+ PDF 原文/论文/来源/报告面板（右）

报告与资料库（/paper/.../run/... · /library）
  对话中生成的报告 → runs / run_outputs → 分屏报告页 · /library 论文与报告库、Snap 对比矩阵 · Zotero & Markdown 导出
  知识库（可选，配置 DIFY_API_BASE 后启用）：/library 内的语料检索与跨文献问答
```

## 技术栈

- **后端**：Python 3.11+ · FastAPI · LangGraph · Deep Agents · PostgreSQL 17 + pgvector（`psycopg` 3 连接池）· 双协议多模型网关（OpenAI Responses & Chat Completions）· OpenTelemetry / Arize Phoenix 追踪 · slowapi 限流
- **前端**：Next.js 15（App Router）· React 19 · TypeScript · Tailwind CSS v4 · react-pdf · react-markdown + KaTeX · 中英双语 i18n
- **外部集成**：MinerU（PDF 版面解析）· Semantic Scholar / OpenAlex / arXiv / Crossref / IEEE Xplore / DBLP / OpenReview（学术检索）· Tavily / Exa / Firecrawl（网页检索与抓取）· EasyScholar（期刊分级）· Dify（知识库 RAG）· Cloudflare R2 / S3（导出配图托管）

## 项目结构

```
scholar/
├── backend/                    # FastAPI + LangGraph + Deep Agents 后端
│   ├── app/
│   │   ├── agents/             # 阅读智能体：harness、模型工厂、TurnGuard、上下文/证据别名中间件、提示词与 24 个工具
│   │   ├── api/                # HTTP & SSE 路由：agent / papers / runs / auth / library / phoenix / admin / system
│   │   ├── db/                 # PostgreSQL 连接池门面、增量 SQL 迁移（0001–0010）与仓储层
│   │   ├── models/             # Pydantic 数据契约（Agent、Evidence、PaperIR、API Schema）
│   │   ├── services/           # 智能体执行与恢复、报告行与结构化视图、模型网关与 Token 计量、MinerU 解析、证据快照、
│   │   │                       #   账号与 OAuth、数据保留期、paper_search / web_search / publication_rank
│   │   └── workflows/          # Sphere 文献地图的检索与筛选步骤、Snap 盲审与外部信号（报告由对话写成）
│   ├── evals/                  # 离线评测框架、LLM 裁判、黄金数据集（reading_seed / chains_seed）与基线
│   ├── scripts/                # 网关/协议/多轮/崩溃恢复验证与运维脚本
│   ├── tests/                  # 后端单元测试套件
│   ├── models.example.toml     # 扩展对话网关与模型注册示例
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/                   # Next.js 15 + React 19 前端
│   ├── src/
│   │   ├── app/                # 路由页面：/chat、/projects、/account、/library、/paper/.../run/...（/upload、/compare 仅作重定向）
│   │   ├── components/         # agent（三栏对话壳/时间线/上下文面板）、snap、lens、sphere、project、account 与公共组件
│   │   ├── hooks/              # 对话事件流（useAgentStream）与持久化布局状态
│   │   └── lib/                # 事件归并器、回合分组、安全链接过滤、证据缓存、API 客户端与双语 i18n
│   ├── tests/                  # 前端单元测试（Node test runner）
│   ├── Dockerfile
│   └── package.json
├── scripts/
│   └── dev.sh                  # 本地前后端一键开发启动脚本
├── docker-compose.yml
└── .env.example
```

## 许可证

本项目以 [MIT 许可证](./LICENSE) 开源。
