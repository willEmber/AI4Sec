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
- 🎯 **三种阅读模式**：Insight Snap 速览要点、Logic Lens 深读公式算法、Research Sphere 梳理文献网络。
- ⚡ **流式输出**：通过 SSE 实时推送分析进度与结果，长文档无需干等。
- 🌐 **多平台检索**：聚合 arXiv、OpenAlex、Semantic Scholar、Crossref、IEEE Xplore 等学术源。
- 📊 **期刊分级**：集成 EasyScholar，展示 SCI / CCF / CSCD 分区，辅助判断文献质量。

## 快速开始（Docker）

### 前置要求

- Docker >= 24.0
- Docker Compose >= 2.20

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 API Key 和模型名称
```

`.env` 中的必填变量：

| 变量 | 说明 |
|---|---|
| `LLM_BASEURL` | OpenAI 兼容 API 地址 |
| `LLM_APIKEY` | LLM API Key |
| `THINKING_MODELNAME` | 推理模型名称 |
| `EMBED_MODELNAME` | 向量模型名称 |
| `RERANK_MODELNAME` | 重排序模型名称 |
| `MINERU_TOKEN` | MinerU PDF 解析 API Token |
| `EASYSCHOLAR_SECRET_KEY` | PublicationRank 使用的 EasyScholar API Key |

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
```

### 前端（Next.js）

```bash
cd frontend
npm install
npm run dev
```

## 架构概览

```
上传 PDF → SHA1 paper_id → 存储 → MinerU API 解析 → content_list.json → PaperIR
    → 按模式路由 → Insight Snap / Logic Lens / Research Sphere
    → 带页码引用的 LLM 分析 → Markdown + LaTeX → SSE 推送到前端
    → 分屏视图：左侧渲染 Markdown，右侧 PDF 查看器支持引用跳页
```

### 阅读模式

- **Insight Snap**：快速生成论文核心洞察的结构化概览。
- **Logic Lens**：深入分析公式、算法和表格。
- **Research Sphere**：探索参考文献网络并识别研究空白。

## 项目结构

```
scholar/
├── backend/             # FastAPI + LangGraph 后端
│   ├── app/
│   │   ├── api/         # REST 接口（papers、runs）
│   │   ├── db/          # SQLite 异步封装
│   │   ├── models/      # Pydantic 模型
│   │   ├── services/    # MinerU、LLM、PaperIR 等服务
│   │   └── workflows/   # LangGraph 子图
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/            # Next.js 15 + React 19 前端
│   ├── src/
│   │   ├── app/         # 页面（首页、上传页、结果页）
│   │   ├── components/  # MarkdownRenderer、PdfViewer、SplitPane
│   │   ├── hooks/       # useRunStream（SSE）
│   │   └── lib/         # API 客户端、类型、国际化
│   ├── Dockerfile
│   └── package.json
├── paper_search/        # 异步多平台论文搜索聚合器
├── papersdownload/      # DOI 到 PDF 的批量下载工具
├── PublicationRank/     # EasyScholar 期刊排名客户端
├── paper_converter/     # MinerU PDF 解析集成
├── docker-compose.yml
├── .env.example
└── CLAUDE.md
```
