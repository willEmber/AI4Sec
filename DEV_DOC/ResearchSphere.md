## 1) Research Sphere 的“交付物”应该长什么样

Research Sphere 不应该只是“更多论文的列表”，而是一个**可操作的研究空间**。建议最终输出至少包含 6 类内容（前端也很好做成分区卡片）：

1. **领域地图（Landscape Map）**
    
    - 把相关工作聚成 3–8 个主题簇（方法范式 / 任务设置 / 数据集路线 / 理论派 vs 工程派）
        
2. **谱系与时间线（Lineage & Timeline）**
    
    - “这篇 paper 从谁来 → 影响了谁 → 哪些分支后续发展”
        
3. **对比矩阵（Comparison Matrix）**
    
    - 中心论文 vs Top-K 竞品/关键引用：核心假设、模型结构、训练目标、数据集、指标、代价（算力/数据）、优势、失败点
        
4. **关键枢纽论文（Hubs）**
    
    - 在你扩展出的子图里：哪些论文是“必读节点”（高中心性、连接多个簇）
        
5. **研究缺口 & 可写点（Gaps & Ideas）**
    
    - 输出“缺口/机会点清单”，并给出**对应证据**来自哪些论文（不是拍脑袋）
        
6. **阅读路线（Reading Path）**
    
    - 3 条路线：快速补课（2–3 篇）、深入复现（3–5 篇）、追前沿（5–10 篇）
        

> 核心原则：Research Sphere 的价值 = **“结构化对比 + 研究路线 + 可证据回溯”**，而不是“更长的总结”。

---

## 2) 输入与可配置项（必须做“预算控制”，否则 Sphere 会失控）

Research Sphere 的扩展是指数型的，所以要把“半径/预算”做成明确参数（默认值给一套能跑通的）：

- **扩展半径（radius）**：
    
    - r=1：只拉 1-hop（参考文献 + 被引 + related/recommendation）
        
    - r=2：再拉 2-hop（用于找分支祖师爷/演化路径）
        
- **候选上限（candidate_cap）**：例如 200（只拉元数据不下载）
    
- **精读上限（pdf_parse_cap）**：例如 10–20（只对最值得的下载并 MinerU 解析）
    
- **LLM 预算（llm_budget）**：每篇只做 Snap/只做 Lens/只做抽取（可选）
    

这样你就能做到：**先快扩展 → 再筛选 → 再回读**，而不是一上来就把 200 篇都下载解析。

---

## 3) 核心数据结构：SphereGraph（用 SQLite 也能跑）

你已经统一用 MinerU 解析 PDF。Research Sphere 需要额外维护一个“围绕中心论文的小图谱”。

### 3.1 Node（论文节点）建议字段

- `node_id`：内部 id（可用 hash）
    
- `canonical_ids`：doi / arxiv_id / openalex_id / s2_paper_id（可为空）
    
- `title, year, venue, authors`
    
- `abstract_text`（如果有）/ `abstract_inverted_index`（OpenAlex 形式）
    
- `cited_by_count`（从 OpenAlex/SS 拿）
    
- `pdf_path`（若已下载）
    
- `mineru_parsed`（bool）
    

> OpenAlex 的 work 对象里，摘要常以 `abstract_inverted_index` 形式出现而非纯文本，需要你解码后再做 embedding/关键词抽取。 ([GitHub](https://github.com/ourresearch/openalex-docs/blob/main/api-entities/works/work-object/README.md?utm_source=chatgpt.com "openalex-docs/api-entities/works/work-object/README.md at main ..."))

### 3.2 Edge（关系边）至少要有 3 类

- `CITES`：A → B（参考文献/引用）
    
- `CITED_BY`：B → A（入边）
    
- `RELATED`：平台推荐/语义相关（OpenAlex related_works、Semantic Scholar recommendations 等） ([OpenAlex](https://docs.openalex.org/api-entities/works?utm_source=chatgpt.com "Works | OpenAlex technical documentation"))
    

你也可以逐步加两类“研究更有用的边”（v2 做）：

- **Bibliographic Coupling**（共引参考文献）：两篇 paper 的参考文献交集越大，越像同一研究路线
    
- **Co-citation**（共被引）：两篇 paper 被同一批后续论文同时引用，往往属于同一经典簇
    

这些边可以完全在你抓到的局部子图里计算，不依赖外部数据库。

---

## 4) 候选论文怎么来：三路并行生成（PDF 内部 + 外部图谱 + 主动检索）

Research Sphere 最强的做法是：**把“引用链（citation chaining）”和“语义检索（semantic search）”结合起来**。

### 4.1 从用户上传 PDF 内部挖“确定性线索”

你强制用 MinerU，所以这里非常顺滑：

- 用 MinerU 的结构化输出（尤其是 content_list / middle.json）定位 **References/Bibliography** 区域，抽每条参考文献文本
    
- MinerU 的输出块天然带 `page_idx` 和 `bbox`，非常适合你做“引用可回溯定位”。 ([OpenDataLab](https://opendatalab.github.io/MinerU/zh/reference/output_files/?utm_source=chatgpt.com "输出文件格式 - MinerU"))
    

> 这一条的价值：PDF 内部的 references 是最可信的“后验真实链接”，用它做 seed，比纯搜索稳定。

### 4.2 用 OpenAlex 做“引用链扩展”（强推荐做你的主 backbone）

OpenAlex 的 Works 实体天然提供：

- `referenced_works`：出边引用（A 引用谁）
    
- `cited_by_api_url`：入边被引（谁引用 A）
    
- `related_works`：相关工作推荐 ([OpenAlex](https://docs.openalex.org/api-entities/works?utm_source=chatgpt.com "Works | OpenAlex technical documentation"))
    

这三个字段就足够让你构建一个 Sphere 的“图骨架”。

### 4.3 用 Semantic Scholar 补强（尤其适合：推荐、元数据补全、PDF 链接）

Semantic Scholar Graph API 提供 paper 的：

- `/graph/v1/paper/{paper_id}/references`
    
- `/graph/v1/paper/{paper_id}/citations`  
    并且有 Recommendations API（基于论文推荐相似论文）。 ([语义学者](https://semanticscholar.readthedocs.io/en/stable/api.html?utm_source=chatgpt.com "API Endpoints - semanticscholar"))
    

它常常比 OpenAlex 更容易拿到可用摘要、关键词或 PDF URL（视具体论文而定），用于补齐“可读信息”。

### 4.4 Crossref 做 DOI/出版信息补全（尤其当 PDF 里没 DOI）

Crossref REST API 是你做“标题/作者/年份 → DOI/期刊信息”的重要兜底。 ([www.crossref.org](https://www.crossref.org/documentation/retrieve-metadata/rest-api/?utm_source=chatgpt.com "REST API - Crossref"))

### 4.5 你现有 paper_search：做“主动检索通道”

当引用链不够（比如论文很新或领域很散），你可以：

- 从 MinerU 抽出的 title/abstract/keywords 生成 query
    
- 用你 `paper_search` 同时打 OpenAlex/arXiv/S2/Crossref/IEEE
    
- 把结果当作 `RELATED` 的候选池
    

**建议把候选来源打上 tag**：`seed_ref / openalex_related / s2_reco / query_search / cited_by`，后面排序与解释都会更清晰。

---

## 5) 候选怎么筛：一个“多目标评分 + 多样性约束”的选择器

Research Sphere 的关键不是“能拉多少”，而是“拉对 + 拉全 + 不重复”。

### 5.1 基础评分（建议可解释、可调权重）

每个候选论文一个分数：

- `S_text`：与中心论文的文本相似度（标题/摘要/关键词）
    
- `S_graph`：在局部引用子图里的重要性（比如 PageRank）  
    PageRank 的定义和用途在 NetworkX 文档里很明确：用于根据入链结构给节点排名。 ([NetworkX](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html?utm_source=chatgpt.com "pagerank — NetworkX 3.6.1 documentation"))
    
- `S_time`：时间衰减（想找前沿就偏向新）
    
- `S_venue`：PublicationRank 输出（如果你要纳入）
    
- `S_novelty`：与已选集合的“差异度”（避免全是同一簇）
    

最终类似：  
`Score = w1*S_text + w2*S_graph + w3*S_time + w4*S_venue + w5*S_novelty`

### 5.2 多样性约束（强烈建议）

否则你会得到 20 篇都讲同一个方法变体。

两种简单可落地策略：

1. **簇后采样**：先聚类（embedding 或关键词）→ 每簇取 Top-n
    
2. **MMR**（最大边际相关）：逐个加入时惩罚与已选论文太相似的候选（实现很简单）
    

---

## 6) “只在必要时下载解析”：分层阅读策略（Sphere 必须这样做）

Research Sphere 要同时满足“广度”和“深度”，最稳的做法是三层：

### Layer 0：元数据层（快）

对所有候选（例如 200 篇）只做：

- 元数据汇总（title/year/venue/cited_by_count）
    
- 1–2 句“为什么它在 Sphere 里”（来自：它引用/被引/related/高中心性/高相似度）
    

### Layer 1：摘要层（中）

对 Top 30（或 50）做：

- abstract 级别的 Snap（方法/贡献/结论 3–5 条）
    
- 抽取：任务、数据集、指标、方法关键词（用于聚类与对比表）
    

> 注意：OpenAlex 的 abstract 可能需要从 `abstract_inverted_index` 解码。 ([GitHub](https://github.com/ourresearch/openalex-docs/blob/main/api-entities/works/work-object/README.md?utm_source=chatgpt.com "openalex-docs/api-entities/works/work-object/README.md at main ..."))

### Layer 2：全文层（慢但价值最高）

对 Top 10–20 才做：

- 调你的 `papersdownload` 下载 PDF
    
- **统一用 MinerU 解析**（你要求的关键点）
    
- 从 MinerU 的 content_list/middle.json 抽：
    
    - 方法细节 / 关键实验表格 / 关键消融 / 失败案例描述  
        并保留 page_idx+bbox 供前端证据跳转。 ([OpenDataLab](https://opendatalab.github.io/MinerU/zh/reference/output_files/?utm_source=chatgpt.com "输出文件格式 - MinerU"))
        

你已经有一个很完整的 MinerU 官方 API 批处理脚本（含轮询、下载 zip、清洗、断点续跑、日志），直接可以复用到 Sphere 的“批量解析邻居论文”步骤中。

---

## 7) 本地检索怎么做：SQLite FTS5 先顶住（再可选向量）

你说存储简化用 SQLite，那 Sphere 的“跨论文检索”可以这样落地：

### 7.1 SQLite FTS5：做块级全文检索（强推荐）

- 给每篇论文的 `blocks`（MinerU content_list 展开）建 FTS5 虚拟表
    
- 你就可以在 Sphere 合成阶段做：
    
    - “在所有邻居论文里搜 _ablation / limitation / dataset split / hyperparameter_”
        
    - “找哪几篇都提到了同一个 benchmark”
        

FTS5 官方文档给了创建虚拟表的标准方式与使用方式。 ([SQLite](https://sqlite.org/fts5.html?utm_source=chatgpt.com "SQLite FTS5 Extension"))

### 7.2 可选：本地向量检索（v2）

如果你觉得“语义聚类/相似检索”需求变强，可以加 FAISS（仍是本地，不引入服务端组件）：

- FAISS 是高效向量相似检索与聚类库（C++/Python）。 ([GitHub](https://github.com/facebookresearch/faiss?utm_source=chatgpt.com "GitHub - facebookresearch/faiss: A library for efficient similarity ..."))
    

> 但 MVP 我建议先用：标题/摘要 embedding + 朴素余弦（几十篇规模足够快），别一上来引入复杂索引。

---

## 8) Sphere 的“合成器”：让 LLM 做它擅长的，不要让它瞎编

Research Sphere 的 LLM 工作最好拆成**三个角色**：

1. **Classifier（聚类与主题命名）**  
    输入：每篇论文的抽取要素（任务/方法/数据集/指标）  
    输出：主题簇 + 每簇一句话定义
    
2. **Comparator（对比表生成）**  
    输入：中心论文 + Top-K（摘要层或全文层抽取结果）  
    输出：结构化对比矩阵（尽量 JSON）
    
3. **Research Advisor（研究机会点生成）**  
    输入：主题 输出：机会点列表（每条对应“证据来源论文/段落”）
    

这能最大限度降低“LLM凭空补细节”的概率，因为它拿到的是你已经抽取过的结构化信息，而不是整篇长文。

---

## 9) LangGraph 子图怎么画（Research Sphere 子图建议）

Research Sphere 本质上是一个“可恢复、可扩展、可中断”的长流程，所以你应该把它做成 **LangGraph 的一个子图**，并用 SQLite checkpoint 保存状态。

- `langgraph-checkpoint-sqlite` 提供 SQLite 实现的 checkpoint saver，非常契合你“存储简单”的目标。 ([PyPI](https://pypi.org/project/langgraph-checkpoint-sqlite/?utm_source=chatgpt.com "langgraph-checkpoint-sqlite · PyPI"))
    

### 推荐节点（从快到慢、层层深入）

1. `sphere_init_from_pdf`
    
    - MinerU parse（中心论文）→ 结构化 blocks
        
2. `extract_core_metadata`
    
    - title/authors/year/doi（如果有）
        
3. `resolve_canonical_ids`
    
    - paper_search + Crossref/SS/OpenAlex 补齐 ID ([www.crossref.org](https://www.crossref.org/documentation/retrieve-metadata/rest-api/?utm_source=chatgpt.com "REST API - Crossref"))
        
4. `expand_graph_candidates`（并行）
    
    - OpenAlex：referenced / cited_by / related ([OpenAlex](https://docs.openalex.org/api-entities/works?utm_source=chatgpt.com "Works | OpenAlex technical documentation"))
        
    - Semantic Scholar：references/citations/reco ([语义学者](https://semanticscholar.readthedocs.io/en/stable/api.html?utm_source=chatgpt.com "API Endpoints - semanticscholar"))
        
    - query_search：paper_search 主动检索
        
5. `dedup_and_score`
    
    - 去重 + 打分 + 多样性选择
        
6. `layer0_summarize_metadata`（全量快）
    
7. `layer1_abstract_snap`（Top-N 中等）
    
8. `download_and_mineru_parse_neighbors`（Top-K 慢）
    
    - 可以调用你现成的 MinerU 批处理脚本跑一个 batch
        
    - MinerU CLI 参数（backend/method/lang/formula/table）也可以做成可配置项。 ([OpenDataLab](https://opendatalab.github.io/MinerU/zh/usage/cli_tools/?utm_source=chatgpt.com "命令行工具 - MinerU"))
        
9. `synthesize_landscape`
    
    - 聚类、命名、输出时间线/谱系、对比矩阵、研究机会点
        
10. `render_markdown`
    
    - JSON → Markdown（供前端渲染）
        

---

## 10) 你可以新增开发的“高杠杆模块”（建议优先级）

按“投入产出比”排序：

### P0：Reference Resolver（参考文献解析与 DOI 归一化）

- 输入：MinerU 抽到的 ref_text 列表
    
- 输出：结构化 ref（title/authors/year/venue/doi/arxiv）
    
- 做法：正则 + Crossref 查询补齐 + 置信度评分 ([www.crossref.org](https://www.crossref.org/documentation/retrieve-metadata/rest-api/?utm_source=chatgpt.com "REST API - Crossref"))
    

### P0：Graph Selector（多目标 + 多样性采样）

- Sphere 的质量很大程度由“你选了哪 20 篇”决定
    
- 做成独立模块，后面随时调权重、加特征
    

### P1：Block-level Evidence Aligner（证据对齐器）

- 把对比表/机会点的每条 claim 绑定到具体论文的 page/bbox（MinerU 已提供底座） ([OpenDataLab](https://opendatalab.github.io/MinerU/zh/reference/output_files/?utm_source=chatgpt.com "输出文件格式 - MinerU"))
    

### P1：Local Search（SQLite FTS5）

- 让 Sphere 能做“跨论文证据检索”，非常实用 ([SQLite](https://sqlite.org/fts5.html?utm_source=chatgpt.com "SQLite FTS5 Extension"))
    

### P2：任务/数据集/指标抽取器（特别适合 ML 场景）

如果你的目标用户很偏 ML/AI，建议加入 Papers with Code 生态做增强：

- PapersWithCode 有官方开源组织与数据集/工具仓库，且有 API client 可用（用于拉任务-数据集-SOTA 关联）。 ([GitHub](https://github.com/paperswithcode?utm_source=chatgpt.com "Papers with code · GitHub"))
    

---

## 11) 我建议你定义一个“Research Sphere 输出 JSON Schema”（再渲染 Markdown）

这样前端能做很强的交互，而不是只能展示一段长文本。一个可用的 schema 大概长这样（示意）：

- `sphere_overview`: { `central_paper`, `radius`, `candidate_count`, `parsed_count` }
    
- `themes[]`: { `name`, `definition`, `representatives[]` }
    
- `timeline[]`: { `year`, `papers[]` }
    
- `key_hubs[:contentReference[oaicite:23]{index=23}ub`, `pagerank_score` }
    
- `comparison_table`: { columns..., rows... }
    
- `gaps_and_ideas[]`: { `idea`, `evidence_papers[]`, `risk` }
    
- `reading_paths`: { `fast_track[]`, `deep_track[]`, `frontier_track[]` }
    

---

### 给你一个可直接落地的“默认 Sphere 配置”（MVP）

- radius=1
    
- 候选上限 200（只元数据）
    
- 摘要层 Top 40
    
- 全文解析 Top 12（MinerU）
    
- 输出：主题簇 5 个、对比表 10 篇、阅读路线 3 条、机会点 8 条
    

这套能在“实验探索”阶段提供非常稳定的价值，而且不会把系统拖爆。

---
