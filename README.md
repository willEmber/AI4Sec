<p align="center">
  <img src="scholar.png" alt="Scholar logo" width="160" />
</p>

<h1 align="center">Scholar Platform</h1>

<p align="center">
  全栈学术论文阅读平台。上传 PDF，选择阅读模式，即可获得带证据引用的结构化 AI 分析，并可跳转回 PDF 原文页面。
</p>

<p align="center">
  <a href="./README.en.md">English README</a>
</p>

## 预览

<p align="center">
  <img src="example.png" alt="Scholar 首页预览" width="760" />
</p>

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
