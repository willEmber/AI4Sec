# PaperSearch（Standalone）

从本仓库抽离出来的 **PaperSearch 学术搜索**模块：聚合多个平台（OpenAlex / Semantic Scholar / arXiv / PubMed / Crossref / InfoXMed / IEEE Xplore），并提供可选的 LLM 重排序。

## 目录结构

```
paper_search_standalone/
├── paper_search/          # 核心代码（可直接拷贝进你的项目）
├── examples/              # 最小示例
├── .env.example           # 可配置环境变量清单（复制为 .env）
└── pyproject.toml         # 可选：作为独立包安装
```

## 集成方式（推荐二选一）

### 方式 A：直接拷贝源码目录

把 `paper_search_standalone/paper_search` 整个目录拷贝到你的项目中，然后：

```python
from paper_search import search_papers
```

### 方式 B：作为本地包安装

在你的项目里执行（或用 uv/pip 等等）：

```bash
pip install -e ./paper_search_standalone
```

安装后可用：

```bash
paper-search -q "transformer attention" --platforms "arXiv,OpenAlex,PubMed"
```

## 配置（环境变量）

1) 复制配置模板：

```bash
cp paper_search_standalone/.env.example .env
```

2) 编辑 `.env`（只需配置你用到的能力）：

- **基础可用（无 Key）**：OpenAlex / arXiv / Crossref 通常可直接跑（强烈建议提供邮箱标识）。
- **建议配置**：
  - `PAPERSEARCH_CONTACT_EMAIL` / `PAPERSEARCH_CONTACT_EMAILS`：对 OpenAlex/Crossref 等作为礼貌标识，也有助于限流策略（支持逗号分隔多邮箱）。
  - `PAPERSEARCH_EMAIL_PICK_STRATEGY`：多邮箱选择策略，支持 `round_robin`（默认）、`random`、`first`。
- **按服务单独配置邮箱池（可选）**：
  - OpenAlex：`PAPERSEARCH_OPENALEX_MAILTO` / `PAPERSEARCH_OPENALEX_MAILTOS`
  - Crossref：`PAPERSEARCH_CROSSREF_MAILTO` / `PAPERSEARCH_CROSSREF_MAILTOS`
- **IEEE Xplore（可选）**：
  - `PAPERSEARCH_IEEE_API_KEY`
  - `PAPERSEARCH_IEEE_PER_SECOND_LIMIT`（默认 10）
  - `PAPERSEARCH_IEEE_DAILY_LIMIT`（默认 200）
- **LLM 重排序（可选）**：
  - `PAPERSEARCH_LLM_BASEURL`、`PAPERSEARCH_LLM_APIKEY`
  - `PAPERSEARCH_RERANK_MODELNAME`（启用重排序 `/rerank`）

完整变量清单见：`paper_search_standalone/.env.example`。

## 快速使用

### 1) CLI

```bash
python -m paper_search -q "large language model retrieval" --platforms "OpenAlex,SemanticScholar,arXiv,PubMed"
```

#### 控制返回字段（CLI）

```bash
python -m paper_search \
  -q "large language model retrieval" \
  --platforms "OpenAlex,arXiv,Crossref" \
  --fields "title,doi,url,source_platform"
```

### 2) 作为 Python 库调用

`search_papers(...)` 是异步函数，返回 **JSON 字符串**（便于直接作为 API 返回）。

```python
import asyncio
import json

from paper_search import search_papers


async def main() -> None:
    out = await search_papers(
        query="transformer attention",
        platforms=["OpenAlex", "SemanticScholar", "arXiv", "PubMed", "Crossref", "InfoXMed", "IEEE Xplore"],
        final_limit=10,
        # 可选：控制返回字段（仅返回你关心的字段，减少输出体积）
        # fields=["title", "doi", "url", "source_platform"],
    )
    data = json.loads(out)
    print(len(data))


if __name__ == "__main__":
    asyncio.run(main())
```

## 平台与行为说明

### 支持的平台名（大小写不敏感）

- `OpenAlex`
- `SemanticScholar`
- `arXiv`
- `PubMed`
- `Crossref`
- `InfoXMed`
- `IEEE Xplore`

### 去重与排序

- **去重**：优先按 DOI，其次按标题指纹（title fingerprint）。
- **排序**：
  - 若配置了 `PAPERSEARCH_RERANK_MODELNAME` 且 LLM 可用：会调用 OpenAI-compatible 的 `/rerank` 做重排序。
  - 否则使用简单词法排序（基于 query 在 title/abstract 的命中）。

### 已移除字段

`pdf_url`、`oa_paper_url`、`agent_remark` 当前版本暂不返回。
下载流程已由独立模块（如 `papersdownload`）负责。

## 输出字段（每条论文）

可选字段名（用于 `fields` / CLI `--fields`）：

- `title`
- `abstract`
- `url`
- `doi`
- `authors`
- `source_platform`

- `title`
- `abstract`
- `url`：落地页链接
- `doi`
- `authors`：`A; B; C` 形式
- `source_platform`：来源平台名
