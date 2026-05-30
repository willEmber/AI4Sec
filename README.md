<p align="center">
  <img src="scholar.png" alt="Scholar logo" width="160" />
</p>

<h1 align="center">Scholar Platform · 学术论文精读平台</h1>

<p align="center">
  上传一篇 PDF，选择阅读模式，即可获得<strong>带证据引用</strong>的结构化 AI 解读——<br/>
  每一条结论都标注页码，一键跳回 PDF 原文核验，让 AI 阅读"有据可查"。
</p>

<p align="center">
    <a href="https://linux.do/t/topic/2108966/20" alt="LINUX DO">
        <img src="https://img.shields.io/badge/LINUX-DO-FFB003.svg?logo=data:image/svg%2bxml;base64,DQo8c3ZnIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAiIHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIj48cGF0aCBkPSJNNjguMi0uMDU1aDYuMjVxMjMuOTY5IDIuMDYyIDM4IDIxLjQyNmM1LjI1OCA3LjY3NiA4LjIxNSAxNi4xNTYgOC44NzUgMjUuNDV2Ni4yNXEtMi4wNjQtMjMuOTY4LTIxLjQzIDM4LTExLjUxMiA3Ljg4NS0yNS40NDUgOC44NzRoLTYuMjVxLTIzLjk3LTIuMDY0LTM4LjAwNC0yMS40M1EuOTcxIDY3LjA1Ni0uMDU0IDUzLjE4di02LjQ3M0MxLjM2MiAzMC43ODEgOC41MDMgMTguMTQ4IDIxLjM3IDguODE3IDI5LjA0NyAzLjU2MiAzNy41MjcuNjA0IDQ2LjgyMS0uMDU2IiBzdHlsZT0ic3Ryb2tlOm5vbmU7ZmlsbC1ydWxlOmV2ZW5vZGQ7ZmlsbDojZWNlY2VjO2ZpbGwtb3BhY2l0eToxIi8+PHBhdGggZD0iTTQ3LjI2NiAyLjk1N3EyMi41My0uNjUgMzcuNzc3IDE1LjczOGE0OS43IDQ5LjcgMCAwIDEgNi44NjcgMTAuMTU3cS00MS45NjQuMjIyLTgzLjkzIDAgOS43NS0xOC42MTYgMzAuMDI0LTI0LjM4N2E2MSA2MSAwIDAgMSA5LjI2Mi0xLjUwOCIgc3R5bGU9InN0cm9rZTpub25lO2ZpbGwtcnVsZTpldmVub2RkO2ZpbGw6IzE5MTkxOTtmaWxsLW9wYWNpdHk6MSIvPjxwYXRoIGQ9Ik03Ljk4IDcwLjkyNmMyNy45NzctLjAzNSA1NS45NTQgMCA4My45My4xMTNRODMuNDI2IDg3LjQ3MyA2Ni4xMyA5NC4wODZxLTE4LjgxIDYuNTQ0LTM2LjgzMi0xLjg5OC0xNC4yMDMtNy4wOS0yMS4zMTctMjEuMjYyIiBzdHlsZT0ic3Ryb2tlOm5vbmU7ZmlsbC1ydWxlOmV2ZW5vZGQ7ZmlsbDojZjlhZjAwO2ZpbGwtb3BhY2l0eToxIi8+PC9zdmc+" alt="LINUX DO" /></a>
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/LangGraph-1C3C3C?logo=langchain&logoColor=white" alt="LangGraph" />
    <img src="https://img.shields.io/badge/Next.js-15-000000?logo=nextdotjs&logoColor=white" alt="Next.js" />
    <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black" alt="React" />
    <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker" />
</p>

<p align="center">
  <a href="./README.en.md">English README</a>
</p>

## 预览

<p align="center">
  <img src="example.png" alt="Scholar 首页预览" width="760" />
</p>

## 核心特性

- 📄 **高精度解析**：基于 MinerU 解析 PDF，完整保留公式、表格、图片与版面层级结构。
- 🔍 **证据可追溯**：每条 AI 结论都附带页码引用，点击即跳回 PDF 原文，杜绝"幻觉"。
- 🎯 **四种阅读模式**：速览要点、深读公式算法、梳理文献网络，以及让 AI 自动选路的智能问答。
- 🧠 **智能问答路由**：直接提问，意图分类器自动路由到最合适的分析路径，或基于单篇论文直接作答。
- 📚 **知识库 RAG**：在你自建的论文语料库上检索与跨文献问答（基于自托管 Dify 检索代理）。
- ⚡ **流式输出**：通过 SSE 实时推送分析进度与结果，长文档无需干等。
- 🌍 **双语输出 + 多模型**：分析流程内部统一用英文，最终结果可翻译为中文/英文；推理模型支持下拉切换。
- 📊 **期刊分级**：集成 EasyScholar，展示 SCI / CCF / CSCD 分区，辅助判断文献质量。

## 阅读模式

| 模式 | 适用场景 |
|---|---|
| **Insight Snap**（快速洞察） | 30 秒速览：核心贡献、关键发现、是否值得深读 |
| **Logic Lens**（逻辑透镜） | 深度分析：公式、算法、实验复现检查清单 |
| **Research Sphere**（研究全景） | 参考文献网络、引用图谱、多论文对比、研究空白，并结合知识库匹配相关工作 |
| **Smart Q&A**（智能问答） | 直接提问，AI 自动判断意图（速览/精读/全景/单篇问答）并路由到最佳路径，或直接给出答案 |

## 知识库（Knowledge Base）

在前端「知识库」页面，可在你自建的论文语料库上进行：

- **检索（Search）**：在整个语料库内全文 / 语义 / 混合检索，支持文档预览与原文跳转。
- **问答（Ask）**：跨多篇论文提问，由 LLM 基于检索片段综合作答并给出来源。

该功能由自托管的 Dify 知识库检索代理提供支撑。设置 `DIFY_API_BASE` 即可启用；留空则自动关闭「知识库」页面与 Research Sphere 中的「库内相关工作」匹配，其余功能不受影响。

## 快速开始（Docker）

### 前置要求

- Docker >= 24.0
- Docker Compose >= 2.20

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 API Key 和模型名称
```

**必填变量**（核心解析与分析）：

| 变量 | 说明 |
|---|---|
| `LLM_BASEURL` | OpenAI 兼容 API 地址 |
| `LLM_APIKEY` | LLM API Key |
| `THINKING_MODELNAME` | 推理模型名称（逗号分隔可填多个，第一个为默认，前端下拉切换） |
| `MINERU_TOKEN` | MinerU PDF 解析 API Token |

**可选变量**（按需开启对应功能）：

| 变量 | 启用的功能 |
|---|---|
| `EMBED_MODELNAME` / `RERANK_MODELNAME` | 问答 / 检索的向量召回与重排序 |
| `EASYSCHOLAR_SECRET_KEY` | 期刊分级（SCI / CCF / CSCD 分区） |
| `TAVILY_KEY` | 期刊分级的 Web 兜底检索 |
| `DIFY_API_BASE` | 知识库 RAG 与 Sphere 库内匹配（见上文） |
| `UNPAYWALL_EMAIL` / `CORE_API_KEY` / `ELSEVIER_API_KEY` / `ELSEVIER_INSTTOKEN` / `WILEY_TDM_TOKEN` | Research Sphere 抓取参考文献全文 |
| `ADMIN_API_TOKEN` | 为 `/api/admin/*` 启用 `X-Admin-Token` 鉴权 |
| `ENABLE_DOCS` | 生产环境设为 `false`，关闭 Swagger / OpenAPI |

> 完整变量与默认值见 [`.env.example`](./.env.example)。

### 2. 构建并启动

```bash
docker compose up -d
```

- 前端：http://localhost:3001
- 后端 API：http://localhost:8001
- API 文档：http://localhost:8001/docs

### 3. 停止服务

```bash
docker compose down
```

### 数据持久化

上传的 PDF 和 SQLite 数据库会持久化到宿主机的 `./docker-data/` 目录。首次启动时该目录会自动创建。

### Docker 镜像源（中国大陆）

Dockerfile 默认使用 `docker.1ms.run` 作为镜像源。若要使用其他镜像源或 Docker Hub，可在构建前设置 `REGISTRY_MIRROR`：

```bash
# 使用其他镜像源
REGISTRY_MIRROR=docker.m.daocloud.io docker compose build

# 直接使用 Docker Hub
REGISTRY_MIRROR=docker.io docker compose build
```

### 代码变更后重新构建

```bash
# 重新构建两个服务
docker compose build

# 只重新构建单个服务
docker compose build backend
docker compose build frontend

# 重新构建并启动
docker compose up -d --build
```

## 本地开发

### 后端（FastAPI）

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000

# 快速导入自检
uv run python -c "from app.main import app; print('OK')"
```

### 前端（Next.js）

```bash
cd frontend
npm install
npm run dev
```

## 架构概览

```
上传 PDF → SHA1 paper_id → 存储 → MinerU 解析 → content_list.json → PaperIR（章节层级 + 块）
    → 按模式路由
        ├─ Insight Snap    速览要点
        ├─ Logic Lens      公式 / 算法 / 实验复现
        ├─ Research Sphere 参考文献网络 + 引用图谱 + 知识库匹配
        └─ Smart Q&A       意图分类 → snap / lens / sphere / qa
    → 带页码引用的 LLM 分析 →（可选）翻译为目标语言 → Markdown + LaTeX
    → SSE 实时推送到前端 → 分屏视图：左侧渲染 Markdown，右侧 PDF 查看器，引用一键跳页
```

## 技术栈

- **后端**：FastAPI · LangGraph 工作流 · 异步 SQLite（aiosqlite）· OpenAI 兼容 LLM 客户端 · slowapi 限流
- **前端**：Next.js 15 · React 19 · TypeScript · Tailwind CSS v4 · react-pdf · react-markdown + KaTeX · 中英双语 i18n
- **外部服务**：MinerU（PDF 解析）· Dify（知识库检索）· EasyScholar（期刊分级）· 多平台学术检索源

## 项目结构

```
scholar/
├── backend/                # FastAPI + LangGraph 后端
│   ├── app/
│   │   ├── api/            # REST 接口：papers / runs / library / system / admin
│   │   ├── db/             # SQLite 异步封装
│   │   ├── models/         # Pydantic 模型（PaperIR、Sphere、API schema）
│   │   ├── services/       # MinerU、LLM、PaperIR、Dify、语料问答、引用图谱、
│   │   │                   #   证据抽取，内置 paper_search / publication_rank
│   │   └── workflows/      # LangGraph 主图 + 子图（snap/lens/sphere/qa）、
│   │                       #   意图分类、翻译、进度推送
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/               # Next.js 15 + React 19 前端
│   ├── src/
│   │   ├── app/            # 页面：首页、上传、结果分屏、知识库
│   │   ├── components/     # MarkdownRenderer、PdfViewer、SplitPane、RankBadges
│   │   ├── hooks/          # useRunStream（SSE）
│   │   └── lib/            # API 客户端、类型、i18n（中/英）
│   ├── Dockerfile
│   └── package.json
├── paper_search/           # 异步多平台论文搜索聚合器（独立 CLI）
├── papersdownload/         # DOI 到 PDF 的批量下载工具（独立 CLI）
├── PublicationRank/        # EasyScholar 期刊排名客户端（独立模块）
├── paper_converter/        # MinerU PDF 解析集成（独立模块）
├── docker-compose.yml
├── .env.example
└── CLAUDE.md
```

> `paper_search/`、`papersdownload/`、`PublicationRank/`、`paper_converter/` 既是可独立运行的命令行工具，其核心能力也已集成进后端 `app/services/` 供全栈应用复用。

## 许可证

本项目以 [MIT 许可证](./LICENSE) 开源。
