# PapersDownload（下载模块）说明文档

本模块用于**按 DOI 批量下载论文 PDF**，并提供 CLI / Python API / GUI（可选）三种使用方式。

> 提示：TDM（Elsevier/Wiley）属于“授权下载”通道，需你自己申请并配置 Token/Key；Sci-Hub 属于非授权来源，默认保留但可随时禁用，使用前请自行评估合规性。

## 1. 功能概览

### 1.1 支持的下载策略（按默认顺序）

每个 DOI 会独立尝试（默认顺序，且可通过参数调整）：

1. **Europe PMC**（开放获取优先）
   - DOI → EuropePMC Search → PMCID → EuropePMC PDF renderer
2. **OA Resolver（Unpaywall / CORE）**
   - 需要配置 `UNPAYWALL_EMAIL` 和/或 `CORE_API_KEY`
3. **Elsevier TDM（ScienceDirect）**
   - 需要 `ELSEVIER_API_KEY`，校外常需 `ELSEVIER_INSTTOKEN`
4. **Wiley TDM（Wiley Online Library）**
   - 需要 `WILEY_TDM_TOKEN`
   - 目前带一个“Wiley DOI 前缀”启发式过滤（常见 `10.1002/`、`10.1111/`），避免对明显非 Wiley DOI 误请求
5. **Sci-Hub 回退（可选）**
   - 作为最后手段尝试多个镜像站

### 1.2 输出与报告

- 默认输出目录：`./pdfs/`
- 默认报告：`./download_report.jsonl`（JSONL，一行一个 DOI 的结果）
- PDF 写入采用 `*.part` 临时文件 + 原子替换，避免中断导致的半文件。

## 2. 凭据（Key/Token/Email）管理

项目**不硬编码**任何敏感信息，统一从：

1) **环境变量**（优先）  
2) 项目根目录本地 `.env` 文件（可选；不会覆盖已有环境变量；且已在 `.gitignore` 中忽略）

### 2.1 `.env` 示例

可直接复制 `.env.example` 为 `.env`：

```bash
UNPAYWALL_EMAIL=mail1@example.com,mail2@example.com
CORE_API_KEY=your_core_api_key
ELSEVIER_API_KEY=your_elsevier_api_key
ELSEVIER_INSTTOKEN=optional_inst_token
WILEY_TDM_TOKEN=your_wiley_tdm_token
```

### 2.2 UNPAYWALL_EMAIL 多邮箱轮询

`UNPAYWALL_EMAIL` 支持多个邮箱轮询（逗号/空格/分号分隔）：

- `UNPAYWALL_EMAIL=a@x.com,b@y.com;c@z.com`

批量下载时会对 DOI 按顺序进行 round-robin 分配邮箱（第 i 个 DOI 用第 `i % N` 个邮箱）。

### 2.3 环境变量清单

- OA Resolver：
  - `UNPAYWALL_EMAIL`（必需其一；可多值轮询）
  - `CORE_API_KEY`（可选；与 Unpaywall 二选一或都配）
- Elsevier TDM：
  - `ELSEVIER_API_KEY`（必需）
  - `ELSEVIER_INSTTOKEN`（可选）
- Wiley TDM：
  - `WILEY_TDM_TOKEN`（必需）

## 3. CLI 使用方式

入口：

- `python3 -m papersdownload ...`
- 或安装后使用 `papersdownload ...`（见 `pyproject.toml` 的脚本入口）

### 3.1 基本用法

```bash
python3 -m papersdownload 10.1038/35057062 10.1111/1755-0998.70015
```

从文件读取（每行一个 DOI，支持 `#` 注释）：

```bash
python3 -m papersdownload --input dois.txt
```

### 3.2 控制并发/超时/重试

```bash
python3 -m papersdownload --workers 8 --timeout 90 --retries 5 --input dois.txt
```

### 3.3 控制策略

- 策略优先级（先 Europe PMC 或先 OA Resolver）：

```bash
python3 -m papersdownload --prefer resolver 10.1038/35057062
```

- 禁用某些回退：

```bash
python3 -m papersdownload --no-resolver --no-elsevier --no-wiley --no-scihub --input dois.txt
```

查看全部参数：

```bash
python3 -m papersdownload --help
```

## 4. Python API 使用方式

### 4.1 批量下载

```python
from papersdownload import download_pdfs

results = download_pdfs(
    ["10.1038/35057062", "10.1111/1755-0998.70015"],
    out_dir="pdfs",
    workers=4,
)
```

### 4.2 结果结构（DownloadResult）

每个结果包含：

- `doi`：规范化后的 DOI
- `ok`：是否成功
- `source`：来源策略（如 `europepmc` / `unpaywall` / `core` / `elsevier_tdm` / `wiley_tdm` / `scihub` / `local`）
- `pdf_path`：成功时的文件路径
- `detail`：补充信息（URL、PMCID 或错误原因）

## 5. GUI（papersdownload_gui.py）

GUI 需要 `PyQt6`。

主要功能：

- 支持多种 DOI 输入格式（文本/多行/JSON/Python 列表等）
- 可勾选策略开关：Europe PMC / OA 解析 / Elsevier TDM / Wiley TDM / Sci-Hub
- 下载完成后可对接 `papersupload` 做文件上传（需要配置 token）

启动示例（Windows PowerShell + uv）：

```powershell
uv run .\papersdownload_gui.py
```

## 6. 常见问题（FAQ）

### 6.1 为什么 Wiley/Elsevier 没生效？

- 未配置对应环境变量（`WILEY_TDM_TOKEN` / `ELSEVIER_API_KEY`）
- 校外访问 Elsevier 需要 `ELSEVIER_INSTTOKEN`
- DOI 并非该出版社：Wiley 会先做前缀过滤（`10.1002/`、`10.1111/`）

### 6.2 下载到的不是 PDF？

下载器会检查 `Content-Type` 或 `%PDF` 文件头；不满足会报 “Not a PDF ...” 并进入重试/回退策略。

