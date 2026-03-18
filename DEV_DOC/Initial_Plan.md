
## 1) 目标与用户主流程（你要的“从 PDF 开始”）

### 1.1 前端用户步骤（唯一主入口）

1. **上传论文 PDF**
    
2. 选择：
    
    - **阅读模式**：Insight Snap / Logic Lens / Research Sphere
        
    - **LLM 模型**：例如“OpenAI / OpenAI-compatible / 本地模型”等（做成可插拔 Provider）
        
    - （可选）**MinerU 解析配置**：backend / method / 语言 / 是否启用公式/表格（做成“高级选项”，默认给一套稳定配置）
        
3. 点击 **开始**
    
4. 前端实时看到进度（解析中 → 建模中 → 生成中 → 完成）
    
5. 结果以 **Markdown + LaTeX** 正确渲染；并且能**跳回 PDF 页码/区域**做“可验证阅读”。
    

---

## 2) 总体架构（最小但可扩展）

### 2.1 组件划分

- **Frontend（React / Next.js 推荐）**
    
    - 上传 PDF、模式/模型选择、进度流式展示、结果 Markdown/LaTeX 渲染、PDF 侧边预览与定位
        
- **Backend（FastAPI + LangGraph）**
    
    - 提供上传接口、任务运行接口、SSE/WebSocket 推流接口
        
    - LangGraph 负责工作流编排：解析 → 结构化 → 路由到阅读模式 → 生成结果
        
- **Parser（MinerU）**
    
    - 强制统一：**所有 PDF（上传的 + 后续下载的）都走 MinerU**
        
    - 两种落地方式二选一（都可行）：
        
        1. **本地 MinerU CLI/服务**（最稳、离线可跑）
            
        2. **MinerU 官方 API**（你已写好脚本基础，直接适配即可）
            
- **Storage**
    
    - SQLite：元数据、任务状态、段落/块索引、产出结果、引用对齐信息
        
    - 本地目录：PDF、MinerU 输出、run 的产物（md/json）
        

---

## 3) MinerU 解析层设计（你系统的“地基”）

> 你要求“入口 PDF 用 MinerU、下载到的 PDF 也用 MinerU”，所以 MinerU 解析层必须抽象成一个统一的 Adapter。

### 3.1 推荐优先：本地 MinerU CLI（MVP 最省事）

MinerU 官方文档给了最直接的命令：`mineru -p <input_path> -o <output_path>`。([opendatalab.github.io](https://opendatalab.github.io/MinerU/zh/usage/quick_usage/ "基础使用 - MinerU"))  
并且 CLI 支持关键解析参数：

- `--method [auto|txt|ocr]`
    
- `--backend [pipeline|hybrid-auto-engine|hybrid-http-client|vlm-auto-engine|vlm-http-client]`
    
- `--lang ...`
    
- `--formula/--table` 等([opendatalab.github.io](https://opendatalab.github.io/MinerU/zh/usage/cli_tools/ "命令行工具 - MinerU"))
    

**建议默认配置（稳定优先）**

- backend：`pipeline` 或 `hybrid-auto-engine`（先稳定提取结构、OCR、表格）
    
- method：`auto`
    
- formula/table：开启（用户可关）
    

> 你前端“选择模型”时，可以把 MinerU backend/method 放在“高级设置”，不干扰主流程。

### 3.2 备选：MinerU 官方 API（已有脚本 → 直接产品化）

你上传的脚本已经把两条路径梳理得很清楚：

- URL 模式：`POST /api/v4/extract/task`，再 `GET /api/v4/extract/task/{task_id}`轮询
    
- 本地批量：`POST /api/v4/file-urls/b:contentReference[oaicite:4]{index=4}`PUT`上传 →`GET /api/v4/extract-results/batch/{batch_id}`轮询并下载 zip
    

工程注意点（避免你踩坑）：

- 批量上传到预签名 URL 时，**PUT 请求不要乱加额外 header（比如 Content-Type）**，否则可能出现 403 签名不匹配；社区也有同类问题说明。([GitHub](https://github.com/opendatalab/MinerU/issues/4145?utm_source=chatgpt.com "https://mineru.net/api/v4/file-urls/batch 返回的下载链接 无法put下载一直返回403"))erU 输出你要怎么用（关键：content_list.json）  
    MinerU 不只输出 markdown，还会输出一组用于“二次开发、质检、定位”的结构化文件。([opendatalab.github.io](https://opendatalab.github.io/MinerU/zh/reference/output_files/ "输出文件格式 - MinerU"))  
    强烈建议你把**content_list.json**作为核心中间表示（比纯 md 更稳定）：
    
- 在 VLM 输出里，content_list 会包含 `type / sub_type / bbox / page_idx` 等字段，并明确支持 code/list/header/footer/page_footnote 等类型（便于清洗、定位、结构化）。([opendatalab.github.io](https://opendatalab.github.io/MinerU/zh/reference/output_files/ "输出文件格式 - MinerU"))
    
- model.json/middle.json 还能给你更底层的块结构、bbox、页面信息等（用于做“点击引用跳 PDF”）。([opendatalab.github.io](https://opendatalab.github.io/MinerU/zh/reference/output_files/ "输出文件格式 - MinerU"))
    

**结论：你的 PaperIR（论文内部表示）建议这样建：**

- `PaperIR.sections[]`：从标题块（title）推断层级结构
    
- `PaperIR.blocks[]`：直接来源于 content_list（带 page_idx + bbox）
    
- `PaperIR.assets[]`：图片、表格、公式（从 block.type 派生）
    
- `PaperIR.refs[]`：参考文献块（ref_text / bibliography 区域）
    

---

## 4) 本地存储方案（SQLite + 本地目录）

### 4.1 本地目录结构（清晰可维护）

建议根目录 `./data`：

```
data/
  app.db                      # SQLite
  papers/
    {paper_id}/
      original.pdf
      mineru/
        raw/                  # MinerU 原始输出（md/json/layout/span 等）
        normalized/           # 你二次结构化后的 PaperIR JSON
      runs/
        {run_id}/
          result.md
          result.json
          logs.jsonl
```

### 4.2 SQLite 表设计（够用 + 可迭代）

**最小可用表：**

1. `papers`
    

- `paper_id`（主键，建议 sha1(pdf_bytes)）
    
- `file_path`
    
- `title`（可后补）
    
- `doi`（可后补）
    
- `created_at`
    

2. `mineru_parses`
    

- `parse_id`
    
- `paper_id`
    
- `backend/method/lang/formula/table`（记录参数，便于复现）
    
- `status`（pending/running/done/failed）
    
- `output_dir`
    

3. `blocks`（来自 content_list）
    

- `block_id`
    
- `paper_id`
    
- `type/sub_type`
    
- `page_idx`
    
- `bbox_json`
    
- `text`（或 md/latex/html 字段）
    
- `section_path`（如 “2.Method/2.1 Model”）
    

4. `runs`
    

- `run_id`
    
- `paper_id`
    
- `mode`（snap/lens/sphere）
    
- `llm_provider` + `llm_model`
    
- `status`
    
- `started_at/finished_at`
    

5. `run_outputs`
    

- `run_id`
    
- `markdown`
    
- `json`（结构化结果，方便前端做卡片/折叠/跳转）
    

**可选增强：**

- `fts_blocks`：SQLite FTS5 做 BM25 检索（替代向量库，仍然“存储简单”）
    
- `citations`：存“结果中的 claim → block/page/bbox”对齐关系，用于可验证跳转
    

---

## 5) LangGraph 工作流（按你的三种阅读模式拆子图）

### 5.1 为什么 LangGraph 很适合你

LangGraph 自带持久化层（checkpointer），**每个 super-step 都会存 checkpoint 到 thread**，支持恢复/重放/人机协作等能力。([LangChain 文档](https://docs.langchain.com/oss/python/langgraph/persistence "Persistence - Docs by LangChain"))  
你这类“解析耗时 + 多阶段推理 + 可能中断重试”的任务，天然适配。

而且你要 SQLite，本身就有现成的 SQLite checkpointer 包：`langgraph-checkpoint-sqlite`（含同步/异步 SqliteSaver/AsyncSqliteSaver）。([PyPI](https://pypi.org/project/langgraph-checkpoint-sqlite/ "langgraph-checkpoint-sqlite · PyPI"))

### 5.2 Graph State 设计（推荐 TypedDict）

核心状态字段建议：

- `paper_id`
    
- `pdf_path`
    
- `mineru_parse_config`
    
- `paper_ir_path`
    
- `mode`
    
- `llm_spec`（provider/model/params）
    
- `intermediate`（抽取的贡献点、公式列表、实验表格等）
    
- `final_markdown` / `final_json`
    

### 5.3 主图节点拆分（MVP 版）

**MainGraph**

1. `ingest_pdf`
    
    - 保存到本地目录
        
    - 写 `papers` 记录
        
2. `mineru_parse`
    
    - 调 MinerU（本地 CLI 或官方 API）
        
    - 写 `mineru_parses` + 落盘 raw 输出
        
3. `build_paper_ir`
    
    - 解析 content_list.json → 写 `blocks`
        
    - 构建 section 层级
        
    - （可选）写 FTS 索引
        
4. `route_by_mode`
    
    - 路由到 Snap / Lens / Sphere 子图
        
5. `assemble_output`
    
    - 统一输出：`result.md` + `result.json`
        
6. `persist_and_finish`
    

### 5.4 三个子图怎么做（对应你的 3 模式）

---

## 6) Insight Snap 子图（浅读：筛选过滤）

**目标：30 秒内判断值不值得精读。**

输入：

- `PaperIR` 的标题、摘要、引言、结论
    
- 贡献点候选（从标题/粗体/列表、以及“we propose / we show / contributions”句式中抽）
    

输出 Markdown 模板（建议固定结构，便于前端卡片化）：

- 论文一句话（问题 + 方法 + 结果）
    
- 3–5 条核心贡献
    
- 关键实验结论（指标/提升幅度/对比对象）
    
- 适用场景 & 局限
    
- “是否值得读”建议（理由）
    
- **证据引用**：每条贡献后面附 `(p.X)` 或 `(p.X, bbox=...)`
    

实现要点：

- 贡献点抽取时，优先用 MinerU 的结构块（title/list/ref_text）而不是纯 regex，稳定性更好。([opendatalab.github.io](https://opendatalab.github.io/MinerU/zh/reference/output_files/ "输出文件格式 - MinerU"))
    

---

## 7) Logic Lens 子图（详细阅读：推导/算法/实验透视）

**目标：深度理解 + 可复现验证。**

输入重点：

- `equation` / `isolate_formula` / `embedding`（行内公式）相关 block
    
- `algorithm` / `code` block（content_list 已支持 code/algorithm sub_type）([opendatalab.github.io](https://opendatalab.github.io/MinerU/zh/reference/output_files/ "输出文件格式 - MinerU"))
    
- 实验表格（table + table_caption + table_footnote）
    
- 方法部分的段落（通过 section_path 定位）
    

输出建议分 6 块：

1. 问题设定（输入输出、假设、符号表）
    
2. 方法总览（流程图式文字 + 关键模块职责）
    
3. 关键公式逐步解释（变量含义、推导逻辑、与 baseline 差异）
    
4. 算法伪代码/步骤复述（逐行解释）
    
5. 实验复现清单（数据集、预处理、训练细节、超参、指标、硬件）
    
6. 可靠性检查（消融是否充分、统计显著性、可能的 confound）
    

工程实现建议：

- 做一个 `Math/Algo Extractor`：从 blocks 里筛出公式/算法/表格，先结构化再喂给 LLM（减少“幻觉总结”）。
    
- 需要“查证”的句子强制附 page_idx/bbox（可做到“点击跳 PDF”）。
    

---

## 8) Research Sphere 子图（深度研究：引用网络 + 竞品对比）

**目标：围绕中心论文构建“球形知识空间”。**

### 8.1 子图步骤

1. `extract_references`
    
    - 从 ref_text / References section 抽参考文献条目（MinerU VLM 的支持类型里含 ref_text）([opendatalab.github.io](https://opendatalab.github.io/MinerU/zh/reference/output_files/ "输出文件格式 - MinerU"))
        
    - 正则提 DOI / arXiv id / venue/year
        
2. `expand_metadata`
    
    - 调用你已有 `paper_search` 聚合多个平台补齐元信息、DOI、URL（这里非常适配你现成工具）
        
3. `rank_and_select_neighbors`
    
    - 选 Top-N：最相关引用、同领域 SOTA、强 baseline
        
    - 排序可用：标题/摘要 embedding、引用次数、与你中心论文关键词重合度等
        
4. `download_pdfs_if_possible`
    
    - 调你已有 `papersdownload`
        
5. **关键：`parse_neighbors_with_mineru`**
    
    - 对下载到的每篇 PDF 走同一套 MinerU parse → blocks 入库（你要求的“后续下载也用 MinerU”在这里落地）
        
6. `compare_and_synthesize`
    
    - 输出对比矩阵：
        
        - 方法假设 / 模型结构 / 训练目标 / 数据集 / 指标 / 结果 / 代码是否开源
            
    - 输出“研究机会点”：差距、可组合模块、未覆盖场景
        

### 8.2 结果呈现建议（前端非常好做）

- 左侧：中心论文
    
- 右侧：相关论文列表（卡片 + 相似度 + 引用关系）
    
- 中间：对比表（可折叠）
    
- 下方：潜在创新点（带证据引用）
    

---

## 9) 前端选型与渲染方案（Markdown + LaTeX 正确渲染）

### 9.1 Markdown 渲染

用 `react-markdown` 做主体渲染（插件生态成熟）。([Remark](https://remarkjs.github.io/react-markdown/?utm_source=chatgpt.com "react-markdown"))

数学公式：

- `remark-math` + `rehype-katex`（LaTeX 渲染）([GitHub](https://github.com/remarkjs/remark-math?utm_source=chatgpt.com "GitHub - remarkjs/remark-math: remark and rehype plugins to support math"))
    

### 9.2 处理 MinerU 可能输出的 HTML（尤其表格）

MinerU 的表格常会以 HTML 片段出现（你从 middle/model/content_list 会遇到），要渲染 HTML：

- 开启 `rehype-raw`（允许 Markdown 内嵌 HTML 被解析）([Npm](https://www.npmjs.com/package/rehype-raw?utm_source=chatgpt.com "rehype-raw - npm"))
    
- 同时建议加 `rehype-sanitize`（避免 raw HTML 带来的注入问题）([GitHub](https://github.com/rehypejs/rehype-sanitize?utm_source=chatgpt.com "GitHub - rehypejs/rehype-sanitize: plugin to sanitize HTML"))
    

> 这不是“合规”，是纯工程安全：否则用户随便传个含恶意 HTML 的 PDF，你前端就可能中招。

### 9.3 PDF 预览与“证据跳转”

- 用 pdf.js / react-pdf 做 PDF viewer
    
- 你在 `blocks` 里存了 `page_idx + bbox` 后：
    
    - summary 里每条结论后放一个小按钮：`[p.5]`
        
    - 点击 → PDF viewer 跳到 page_idx，并在 bbox 画一个半透明高亮框（这会成为你产品的差异化）
        

---

## 10) 把你现有工具接进来（不改变入口，但增强 Sphere）

你现有三件工具都能自然融入：

1. **paper_search**
    

- 用于 Research Sphere 的“补齐引用/找竞品/找后续工作”
    
- 也可用于：从 PDF 解析出的标题去反查 DOI
    

2. **PublicationRank**
    

- Sphere 输出里加一个“发表质量”字段（SCI Q、CCF 等），辅助筛选
    

3. **papersdownload**
    

- Sphere 里选 Top-N 论文下载
    
- **下载后统一走 MinerU 解析入库**（满足你的新要求）
    

---

## 11) 建议的迭代交付顺序（从能跑到好用）

### 阶段 A：MVP（先把闭环跑通）

- 上传 PDF → 本地落盘
    
- MinerU 解析 → 生成 blocks/content_list 入 SQLite
    
- Insight Snap 输出 markdown（带 page_idx 引用）
    
- React 正确渲染 markdown + latex
    

### 阶段 B：Logic Lens（结构化深读）

- 公式/算法/表格抽取器
    
- 复现清单模板 + 可验证引用
    
- 基于 section_path 的“按章节阅读”
    

### 阶段 C：Research Sphere（知识空间）

- ref 抽取 → paper_search 补齐 → 排序
    
- 下载 Top-N → MinerU 解析 → 多论文对比输出
    
- 前端做“对比矩阵 + 引用网络”（先列表化也行）
    

### 阶段 D：体验打磨（让它像产品）

- SSE 流式输出（边生成边展示）
    
- 解析失败重试/断点续跑（LangGraph + SQLite checkpointer）
    
- 历史 run 管理（一个 paper 多次 run，对比不同模型/模式）
    

---

## 12) 你这版方案里最关键的“工程抓手”

1. **把 MinerU 的 content_list.json 当作系统主数据**（比 markdown 稳、可定位、可重建结构）([opendatalab.github.io](https://opendatalab.github.io/MinerU/zh/reference/output_files/ "输出文件格式 - MinerU"))
    
2. **用 SQLite + FTS5 先跑通检索**（足够支撑 Lens/Sphere 的“按需引用块”）
    
3. **LangGraph 用 SQLite checkpointer**，保证耗时流程可恢复、可回放（对解析/下载这种长链路非常关键）([LangChain 文档](https://docs.langchain.com/oss/python/langgraph/persistence "Persistence - Docs by LangChain"))
    
4. 前端把 **“结论 → 证据页/框”** 做出来，你的智能体会立刻从“会总结”变成“可验证阅读助手”。
    

---

## 13) 你给的 MinerU 官方 API 脚本怎么用（建议落地方式）

你上传的脚本已经把：

- API 交互（创建任务/批量上传/轮询/下载 zip）
    
- 清洗逻辑入口（CleanConfig、process_zip_to_clean_md）
    
- 断点续跑 state 文件
    

都搭好了。最省事的做法是：

- 把它拆成 `mineru_adapter/official_api.py`（保留 MinerUClient + poll + download + unzip）
    
- 在 LangGraph 的 `mineru_parse` 节点里调用它
    
- 输出目录直接写到 `data/papers/{paper_id}/mineru/raw/`  
    脚本本体可直接作为参考实现：
    

---

