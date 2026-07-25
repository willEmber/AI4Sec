"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

export type Locale = "en" | "zh";

const translations: Record<Locale, Record<string, string>> = {
  en: {
    // Nav
    "nav.brand": "Scholar",
    "nav.upload": "Upload & Analyze",
    "nav.library": "Knowledge Base",

    // Landing page
    "home.title": "Scholar Platform",
    "home.subtitle":
      "Upload an academic paper and get structured AI-powered analysis with evidence citations linking back to PDF pages.",
    "home.mode.snap.title": "Insight Snap",
    "home.mode.snap.desc":
      "30-second triage. Core contributions, key findings, and whether it\u2019s worth reading.",
    "home.mode.lens.title": "Logic Lens",
    "home.mode.lens.desc":
      "Deep analysis of formulas, algorithms, and experiments with reproduction checklists.",
    "home.mode.sphere.title": "Research Sphere",
    "home.mode.sphere.desc":
      "Reference network analysis with multi-paper comparison and research gap identification.",
    "home.cta": "Upload Paper",
    "home.eyebrow": "AI-powered paper analysis",
    "home.modes_heading": "Three ways to read a paper",
    "home.modes_sub": "Pick a lens, or let Smart Q&A choose for you.",

    // Upload page
    "upload.title": "Upload & Analyze Paper",
    "upload.drop": "Drop PDF here or click to browse",
    "upload.supports": "Supports .pdf files",
    "upload.drop_error": "Please drop a PDF file",
    "upload.mode_label": "Reading Mode",
    "upload.mode.snap.label": "Insight Snap",
    "upload.mode.snap.desc": "30-second triage: contributions, findings, worth-reading assessment",
    "upload.mode.lens.label": "Logic Lens",
    "upload.mode.lens.desc": "Deep analysis: formulas, algorithms, experiment reproduction checklist",
    "upload.mode.sphere.label": "Research Sphere",
    "upload.mode.sphere.desc": "Reference network: multi-paper comparison, research gaps",
    "upload.mode.auto.label": "Smart Q&A",
    "upload.mode.auto.desc": "Ask a question; AI picks the best analysis mode (or answers directly)",
    "upload.question_label": "Your question",
    "upload.question_placeholder": "e.g., What datasets did they use? How does the loss function work?",
    "upload.question_required": "Please enter a question for Smart Q&A mode",
    "upload.question_hint": "AI will detect intent and route to the best analysis path",
    "upload.model_label": "LLM Model",
    "upload.model_loading": "Loading models…",
    "upload.model_empty": "No models configured (set THINKING_MODELNAME)",
    "upload.language_label": "Output Language",
    "upload.language_note": "Research workflow uses English internally; only the final output is translated.",
    "upload.submit": "Start Analysis",
    "upload.submitting": "Uploading & starting analysis...",
    "upload.fail": "Upload failed",

    // Recent runs panel
    "recent.active_header": "{count} run(s) still in progress — click to resume",
    "recent.history_header": "Recent runs ({count})",
    "recent.starting": "Starting...",
    "recent.resumed_toast": "Reconnected — showing latest progress",
    "recent.dismiss": "Dismiss",

    // Run page
    "run.mode": "Mode",
    "run.status.running": "Running...",
    "run.status.complete": "Complete",
    "run.status.failed_label": "Analysis Failed",
    "run.status.unknown": "An unknown error occurred",
    "run.starting": "Starting analysis...",
    "run.export_md": "Export .md",
    "run.export_zotero": "Zotero (.rdf)",
    "run.export_zotero_title": "Download a Zotero import bundle (.zip): item + report note + PDF. Unzip, then File → Import the .rdf into your local Zotero.",
    "run.your_question": "Your question:",
    "run.detected_intent": "Detected intent:",
    "run.pdf_ready_hint": "The PDF is ready on the right — feel free to start reading while the analysis runs.",

    // Intent labels (mirrors mode labels but used for detected_intent display)
    "intent.snap": "Insight Snap",
    "intent.lens": "Logic Lens",
    "intent.sphere": "Research Sphere",
    "intent.qa": "Direct Q&A",

    // Steps
    "step.ingest_pdf": "Verifying PDF",
    "step.mineru_parse": "Parsing with MinerU",
    "step.build_paper_ir": "Building document structure",
    "step.enrich_metadata": "Looking up publication rank",
    "step.classify_intent": "Detecting question intent",
    "step.run_snap": "Generating Insight Snap",
    "step.run_lens": "Generating Logic Lens",
    "step.run_sphere": "Generating Research Sphere",
    "step.run_qa": "Answering your question",
    "step.translate_output": "Translating output",
    "step.persist_output": "Saving results",
    "step.sphere_init_from_pdf": "Extracting references from PDF",
    "step.extract_core_metadata": "Extracting core metadata",
    "step.resolve_canonical_ids": "Resolving paper identifiers",
    "step.expand_graph_candidates": "Expanding citation graph",
    "step.enrich_candidates": "Completing candidate metadata",
    "step.score_and_rank": "Scoring quality and venue ranks",
    "step.relevance_gate": "Judging relevance to this paper",
    "step.build_similarity_graph": "Building similarity graph",
    "step.select_core_set": "Selecting core literature set",
    "step.dedup_and_score": "Scoring and selecting candidates",
    "step.layer0_summarize_metadata": "Summarizing metadata",
    "step.layer1_abstract_snap": "Analyzing abstracts",
    "step.download_and_mineru_parse": "Downloading and parsing papers",
    "step.synthesize_landscape": "Synthesizing research landscape",
    "step.render_output": "Rendering final output",

    // PDF viewer
    "pdf.prev": "Prev",
    "pdf.next": "Next",
    "pdf.loading": "Loading PDF...",
    "pdf.jump_to_page": "Jump to page",
    "pdf.collapse": "Hide PDF",
    "pdf.expand": "Show PDF",

    // Knowledge base (library)
    "library.title": "Knowledge Base",
    "library.subtitle": "Search and ask across your research-paper corpus.",
    "library.tab.search": "Search",
    "library.tab.ask": "Ask",
    "library.method_label": "Retrieval",
    "library.method.full_text_search": "Fast",
    "library.method.semantic_search": "Semantic",
    "library.method.hybrid_search": "Hybrid",
    "library.method_slow_hint": "Semantic / hybrid search is higher quality but slow (may take 30s+).",
    "library.search_placeholder": "Search the corpus…",
    "library.ask_placeholder": "Ask a question across all your papers…",
    "library.run_search": "Search",
    "library.run_ask": "Ask",
    "library.searching": "Searching…",
    "library.thinking": "Thinking…",
    "library.results_count": "{count} result(s)",
    "library.no_results": "No results.",
    "library.documents": "Documents",
    "library.load_more": "Load more",
    "library.sources": "Sources",
    "library.open_doc": "Open source document",
    "library.doc_loading": "Loading document…",
    "library.select_hint": "Select a result or document to preview it here.",
    "library.empty_query": "Enter a search query first.",
    "library.empty_question": "Enter a question first.",
    "library.disabled": "Knowledge base is not configured (set DIFY_API_BASE).",
    "library.error": "Request failed",

    // Research Sphere — structured report view
    // Compare / triage matrix
    "nav.compare": "Compare",
    "matrix.title": "Triage Workbench",
    "matrix.subtitle": "Compare finished Insight Snap runs side by side. Metrics measured by more than one paper line up in their own column.",
    "matrix.loading": "Loading runs\u2026",
    "matrix.empty": "No completed Insight Snap runs yet.",
    "matrix.pick": "Comparing {n} of {total} runs",
    "matrix.select_prompt": "Select at least one run to compare.",
    "matrix.unavailable": "This run has no structured payload and cannot be compared.",
    "matrix.unavailable_n": "{n} run(s) predate the structured report and cannot be compared.",
    "matrix.sort_by": "Sort by",
    "matrix.sort.tier": "Verdict",
    "matrix.sort.citations": "Citations",
    "matrix.sort.year": "Year",
    "matrix.sort.title": "Title",
    "matrix.col.paper": "Paper",
    "matrix.col.verdict": "Verdict",
    "matrix.col.limitation": "Main limitation",
    "matrix.retracted_short": "retracted",
    "matrix.no_shared_metrics": "No metric is reported by more than one of the selected papers, so there are no shared columns to compare.",

    // Insight Snap — structured report
    "snap.view.structured": "Structured",
    "snap.view.markdown": "Markdown",
    "snap.tier.must_read": "Must read",
    "snap.tier.selective": "Selective read",
    "snap.tier.skip": "Can skip",
    "snap.sec.evidence": "Evidence",
    "snap.sec.review": "Content review",
    "snap.sec.worth_time": "Worth your time",
    "snap.sec.contributions": "Core contributions",
    "snap.sec.findings": "Key experimental findings",
    "snap.sec.applicability": "Applicability & limitations",
    "snap.sec.decide": "Deciding",
    "snap.sig.venue": "Venue",
    "snap.sig.citations": "Citation impact",
    "snap.sig.influential": "Influential citations",
    "snap.sig.intents": "Cited as",
    "snap.intent.methodology": "methodology",
    "snap.intent.result": "result",
    "snap.intent.background": "background",
    "snap.intent.sampled": "of {n} sampled",
    "snap.sig.code": "Official code",
    "snap.sig.code_third_party": "Code links",
    "snap.sig.oa": "Open access",
    "snap.sig.retraction": "Retraction",
    "snap.sig.retracted": "\u26a0\ufe0f This paper has been retracted",
    "snap.sig.not_retracted": "None found",
    "snap.sig.preprint": "preprint",
    "snap.sig.survey": "survey",
    "snap.dim.contribution": "Contribution",
    "snap.dim.evidence": "Evidence",
    "snap.dim.novelty": "Novelty",
    "snap.dim.reproducibility": "Reproducibility",
    "snap.dim.limitations": "Limitation severity",
    "snap.col.metric": "Metric",
    "snap.col.dataset": "Dataset",
    "snap.col.value": "This paper",
    "snap.col.baseline": "Baseline",
    "snap.col.delta": "\u0394",
    "snap.col.page": "Page",
    "snap.score.content": "content",
    "snap.score.external": "external",
    "snap.score.combined": "combined",
    "snap.audit.coverage": "{pct}% of claims cited",
    "snap.no_findings": "The paper reports no quantified results in the extracted text.",
    "snap.no_review": "The content review did not complete; this verdict rests on external signals alone.",
    "snap.override.thin_record": "Scored on content alone \u2014 the paper has no citation record yet.",
    "snap.unresolved": "Not matched in OpenAlex or Semantic Scholar, so citation, access and retraction status are unknown rather than zero.",
    "snap.red_flags": "Red flags",
    "snap.limitations": "Limitations",
    "snap.skip_cost": "Cost of skipping",
    "snap.readers": "Who should read it",
    "snap.for_question": "Against your question",
    "snap.open_source": "Open source record",
    "snap.unknown": "unknown",
    "snap.none": "none found",
    "snap.yes": "Yes",
    "snap.no": "No",
    "snap.per_year": "/yr",

    "sphere.view.structured": "Structured",
    "sphere.view.markdown": "Markdown",
    "sphere.view.cards": "Card view",
    "sphere.view.table": "Table view",
    "sphere.this_paper": "This paper",
    "sphere.cited_n": "{n} citations",
    "sphere.influential": "Influential",
    "sphere.stat.core": "Core papers",
    "sphere.stat.candidates": "Candidates screened",
    "sphere.stat.themes": "Themes",
    "sphere.stat.span": "Year span",
    "sphere.sec.overview": "Landscape",
    "sphere.sec.core": "Core literature",
    "sphere.sec.themes": "Themes",
    "sphere.sec.timeline": "Timeline",
    "sphere.sec.comparison": "Comparison",
    "sphere.sec.hubs": "Hubs",
    "sphere.sec.gaps": "Gaps",
    "sphere.sec.paths": "Reading paths",
    "sphere.sec.library": "Your library",
    "sphere.core.intro":
      "{n} papers passed provenance, quality and relevance review, grouped by their relation to this paper.",
    "sphere.core.all": "All",
    "sphere.partition.foundation": "Foundations & Background",
    "sphere.partition.foundation.desc": "Prior work this paper builds on or sets out to replace.",
    "sphere.partition.method": "Method Neighbors & Competitors",
    "sphere.partition.method.desc": "Alternative approaches to the same problem, compared or competing.",
    "sphere.partition.follow_up": "Follow-up Work",
    "sphere.partition.follow_up.desc": "Work that extends, analyses or improves this paper's method.",
    "sphere.partition.application": "Cross-domain Applications",
    "sphere.partition.application.desc": "The idea carried into other tasks and domains.",
    "sphere.partition.frontier": "Recent Frontier",
    "sphere.partition.frontier.desc": "The most recent relevant work, published after this paper.",
    "sphere.partition.survey": "Surveys & Reviews",
    "sphere.partition.survey.desc": "Entry points that map the wider area.",
    "sphere.partition.library": "From Your Library",
    "sphere.partition.library.desc": "Related papers found in your own knowledge base.",
    "sphere.themes.intro": "Communities detected from citation-structure similarity (bibliographic coupling).",
    "sphere.timeline.intro": "How the line of work developed year by year.",
    "sphere.cmp.intro": "{n} papers side by side — tap a row to expand.",
    "sphere.cmp.paper": "Paper",
    "sphere.cmp.problem": "Problem",
    "sphere.cmp.method": "Method",
    "sphere.cmp.dataset": "Dataset",
    "sphere.cmp.metric": "Metric",
    "sphere.cmp.strength": "Strength",
    "sphere.cmp.weakness": "Weakness",
    "sphere.hubs.intro": "Most structurally central papers in the citation graph.",
    "sphere.hubs.pagerank": "PageRank on the similarity graph",
    "sphere.gaps.evidence": "Evidence",
    "sphere.path.fast": "Fast",
    "sphere.path.deep": "Deep",
    "sphere.path.frontier": "Frontier",
    "sphere.path.steps": "{n} papers",
    "sphere.library.intro": "Papers from your own knowledge base that are semantically related.",
  },
  zh: {
    // Nav
    "nav.brand": "Scholar",
    "nav.upload": "上传与分析",
    "nav.library": "知识库",

    // Landing page
    "home.title": "Scholar 学术平台",
    "home.subtitle":
      "上传学术论文，获取结构化AI分析报告，证据引用直接链接到PDF对应页面。",
    "home.mode.snap.title": "快速洞察 (Insight Snap)",
    "home.mode.snap.desc":
      "30秒速览：核心贡献、关键发现、是否值得深读。",
    "home.mode.lens.title": "逻辑透镜 (Logic Lens)",
    "home.mode.lens.desc":
      "深度分析公式、算法与实验，附复现检查清单。",
    "home.mode.sphere.title": "研究全景 (Research Sphere)",
    "home.mode.sphere.desc":
      "参考文献网络分析，多论文对比与研究空白识别。",
    "home.cta": "上传论文",
    "home.eyebrow": "AI 驱动的论文分析",
    "home.modes_heading": "三种论文阅读模式",
    "home.modes_sub": "选择一种分析视角，或交给智能问答自动决定。",

    // Upload page
    "upload.title": "上传与分析论文",
    "upload.drop": "将PDF拖放到此处或点击浏览",
    "upload.supports": "支持 .pdf 文件",
    "upload.drop_error": "请拖放PDF文件",
    "upload.mode_label": "阅读模式",
    "upload.mode.snap.label": "快速洞察 (Insight Snap)",
    "upload.mode.snap.desc": "30秒速览：核心贡献、关键发现、是否值得深读",
    "upload.mode.lens.label": "逻辑透镜 (Logic Lens)",
    "upload.mode.lens.desc": "深度分析：公式、算法、实验复现检查清单",
    "upload.mode.sphere.label": "研究全景 (Research Sphere)",
    "upload.mode.sphere.desc": "参考文献网络：多论文对比、研究空白",
    "upload.mode.auto.label": "智能问答 (Smart Q&A)",
    "upload.mode.auto.desc": "提出问题，AI 自动选择最合适的分析路径（也可直接作答）",
    "upload.question_label": "您的问题",
    "upload.question_placeholder": "如：他们用了什么数据集？损失函数是如何工作的？",
    "upload.question_required": "智能问答模式需要输入问题",
    "upload.question_hint": "AI 将识别问题意图，并选择最合适的分析路径",
    "upload.model_label": "LLM 模型",
    "upload.model_loading": "正在加载模型…",
    "upload.model_empty": "未配置可选模型（请设置 THINKING_MODELNAME）",
    "upload.language_label": "输出语言",
    "upload.language_note": "研究流程内部使用英文；仅最终输出会被翻译。",
    "upload.submit": "开始分析",
    "upload.submitting": "正在上传并启动分析...",
    "upload.fail": "上传失败",

    // 最近运行面板
    "recent.active_header": "{count} 个任务仍在进行中 — 点击恢复查看",
    "recent.history_header": "最近运行（{count}）",
    "recent.starting": "启动中...",
    "recent.resumed_toast": "已重新连接 — 已显示最新进度",
    "recent.dismiss": "清除",

    // Run page
    "run.mode": "模式",
    "run.status.running": "运行中...",
    "run.status.complete": "已完成",
    "run.status.failed_label": "分析失败",
    "run.status.unknown": "发生未知错误",
    "run.starting": "正在启动分析...",
    "run.export_md": "导出 .md",
    "run.export_zotero": "Zotero 导入包",
    "run.export_zotero_title": "下载 Zotero 导入包（.zip）：条目 + 报告笔记 + PDF。解压后在 Zotero 中“文件 → 导入”该 .rdf，即可导入本地库（无需 Zotero 云账户）。",
    "run.your_question": "您的问题：",
    "run.detected_intent": "识别意图：",
    "run.pdf_ready_hint": "右侧已加载论文原文，AI 分析期间可先行阅读。",

    // 意图标签（对应模式名称，仅用于展示分类器识别结果）
    "intent.snap": "快速洞察",
    "intent.lens": "逻辑透镜",
    "intent.sphere": "研究全景",
    "intent.qa": "直接问答",

    // Steps
    "step.ingest_pdf": "验证PDF文件",
    "step.mineru_parse": "MinerU解析中",
    "step.build_paper_ir": "构建文档结构",
    "step.enrich_metadata": "查询期刊等级",
    "step.classify_intent": "识别问题意图",
    "step.run_snap": "生成快速洞察",
    "step.run_lens": "生成逻辑透镜",
    "step.run_sphere": "生成研究全景",
    "step.run_qa": "解答您的问题",
    "step.translate_output": "翻译输出内容",
    "step.persist_output": "保存结果",
    "step.sphere_init_from_pdf": "从PDF提取参考文献",
    "step.extract_core_metadata": "提取核心元数据",
    "step.resolve_canonical_ids": "解析论文标识符",
    "step.expand_graph_candidates": "扩展引用图谱",
    "step.enrich_candidates": "补全候选论文元数据",
    "step.score_and_rank": "评估论文质量与期刊等级",
    "step.relevance_gate": "审查与本文的相关性",
    "step.build_similarity_graph": "构建引文相似图",
    "step.select_core_set": "遴选核心文献集",
    "step.dedup_and_score": "评分与筛选候选论文",
    "step.layer0_summarize_metadata": "汇总元数据",
    "step.layer1_abstract_snap": "分析摘要",
    "step.download_and_mineru_parse": "下载并解析论文",
    "step.synthesize_landscape": "综合研究全景",
    "step.render_output": "渲染最终输出",

    // PDF viewer
    "pdf.prev": "上一页",
    "pdf.next": "下一页",
    "pdf.loading": "加载PDF中...",
    "pdf.jump_to_page": "跳转到第{page}页",
    "pdf.collapse": "收起原文",
    "pdf.expand": "展开原文",

    // 知识库
    "library.title": "我的文献库",
    "library.subtitle": "在你的论文语料库中检索与问答。",
    "library.tab.search": "检索",
    "library.tab.ask": "问答",
    "library.method_label": "检索方式",
    "library.method.full_text_search": "快速",
    "library.method.semantic_search": "语义",
    "library.method.hybrid_search": "混合",
    "library.method_slow_hint": "语义/混合检索质量更高但较慢（可能 30 秒以上）。",
    "library.search_placeholder": "检索语料库…",
    "library.ask_placeholder": "向你的全部论文提问…",
    "library.run_search": "检索",
    "library.run_ask": "提问",
    "library.searching": "检索中…",
    "library.thinking": "生成中…",
    "library.results_count": "{count} 条结果",
    "library.no_results": "无结果。",
    "library.documents": "文献列表",
    "library.load_more": "加载更多",
    "library.sources": "来源",
    "library.open_doc": "打开来源文档",
    "library.doc_loading": "加载文档中…",
    "library.select_hint": "在此预览所选结果或文档。",
    "library.empty_query": "请先输入检索词。",
    "library.empty_question": "请先输入问题。",
    "library.disabled": "尚未配置知识库（请设置 DIFY_API_BASE）。",
    "library.error": "请求失败",

    // 研究全景 —— 结构化报告视图
    // 对比 / 分诊工作台
    "nav.compare": "对比",
    "matrix.title": "分诊工作台",
    "matrix.subtitle": "把已完成的快速洞察结果并排比较。被多篇论文共同测量的指标会自动对齐成独立的一列。",
    "matrix.loading": "正在加载分析记录……",
    "matrix.empty": "还没有已完成的快速洞察分析。",
    "matrix.pick": "已选 {n} / {total} 项",
    "matrix.select_prompt": "请至少选择一项进行对比。",
    "matrix.unavailable": "该记录没有结构化数据,无法参与对比。",
    "matrix.unavailable_n": "有 {n} 项分析早于结构化报告,无法参与对比。",
    "matrix.sort_by": "排序",
    "matrix.sort.tier": "分诊结论",
    "matrix.sort.citations": "引用数",
    "matrix.sort.year": "年份",
    "matrix.sort.title": "标题",
    "matrix.col.paper": "论文",
    "matrix.col.verdict": "分诊结论",
    "matrix.col.limitation": "主要局限",
    "matrix.retracted_short": "已撤稿",
    "matrix.no_shared_metrics": "所选论文中没有任何指标被多篇共同测量,因此没有可对比的共享列。",

    // 快速洞察 —— 结构化报告
    "snap.view.structured": "结构化视图",
    "snap.view.markdown": "Markdown 原文",
    "snap.tier.must_read": "必读",
    "snap.tier.selective": "选读",
    "snap.tier.skip": "可跳过",
    "snap.sec.evidence": "证据",
    "snap.sec.review": "内容评审",
    "snap.sec.worth_time": "值得花时间的部分",
    "snap.sec.contributions": "核心贡献",
    "snap.sec.findings": "关键实验发现",
    "snap.sec.applicability": "适用性与局限",
    "snap.sec.decide": "如何决定",
    "snap.sig.venue": "发表载体",
    "snap.sig.citations": "引用影响",
    "snap.sig.influential": "高影响力引用",
    "snap.sig.intents": "被引用方式",
    "snap.intent.methodology": "方法学",
    "snap.intent.result": "结果",
    "snap.intent.background": "背景",
    "snap.intent.sampled": "抽样 {n} 条",
    "snap.sig.code": "官方代码",
    "snap.sig.code_third_party": "代码链接",
    "snap.sig.oa": "开放获取",
    "snap.sig.retraction": "撤稿状态",
    "snap.sig.retracted": "\u26a0\ufe0f 本文已被撤稿",
    "snap.sig.not_retracted": "未发现",
    "snap.sig.preprint": "预印本",
    "snap.sig.survey": "综述",
    "snap.dim.contribution": "贡献强度",
    "snap.dim.evidence": "证据充分性",
    "snap.dim.novelty": "新颖性",
    "snap.dim.reproducibility": "可复现性",
    "snap.dim.limitations": "局限严重度",
    "snap.col.metric": "指标",
    "snap.col.dataset": "数据集",
    "snap.col.value": "本文",
    "snap.col.baseline": "基线",
    "snap.col.delta": "变化",
    "snap.col.page": "页码",
    "snap.score.content": "内容",
    "snap.score.external": "外部证据",
    "snap.score.combined": "综合",
    "snap.audit.coverage": "{pct}% 陈述已带引用",
    "snap.no_findings": "提取文本中未给出可量化的实验结果。",
    "snap.no_review": "内容评审未能完成,本结论仅基于外部信号。",
    "snap.override.thin_record": "仅按内容评分 —— 该论文尚无引用记录。",
    "snap.unresolved": "未能在 OpenAlex / Semantic Scholar 中匹配到记录,因此引用数、开放获取与撤稿状态为「未知」而非零。",
    "snap.red_flags": "风险提示",
    "snap.limitations": "局限",
    "snap.skip_cost": "跳过的代价",
    "snap.readers": "适合读者",
    "snap.for_question": "针对你的问题",
    "snap.open_source": "查看源记录",
    "snap.unknown": "未知",
    "snap.none": "未发现",
    "snap.yes": "是",
    "snap.no": "否",
    "snap.per_year": "/年",

    "sphere.view.structured": "结构化视图",
    "sphere.view.markdown": "Markdown 原文",
    "sphere.view.cards": "卡片视图",
    "sphere.view.table": "表格视图",
    "sphere.this_paper": "本文",
    "sphere.cited_n": "被引 {n}",
    "sphere.influential": "高影响引用",
    "sphere.stat.core": "核心文献",
    "sphere.stat.candidates": "候选论文",
    "sphere.stat.themes": "主题聚类",
    "sphere.stat.span": "年份跨度",
    "sphere.sec.overview": "研究全景",
    "sphere.sec.core": "核心文献",
    "sphere.sec.themes": "主题聚类",
    "sphere.sec.timeline": "时间脉络",
    "sphere.sec.comparison": "对比矩阵",
    "sphere.sec.hubs": "关键枢纽",
    "sphere.sec.gaps": "研究空白",
    "sphere.sec.paths": "阅读路径",
    "sphere.sec.library": "我的文献库",
    "sphere.core.intro": "以下 {n} 篇论文经来源分层、质量评估与相关性审查后入选核心集，按其与本文的关系分区。",
    "sphere.core.all": "全部",
    "sphere.partition.foundation": "奠基与背景",
    "sphere.partition.foundation.desc": "本文所依托或意在替代的前置工作。",
    "sphere.partition.method": "方法近邻与竞品",
    "sphere.partition.method.desc": "面向同一问题的其他技术路线，构成对比或竞争关系。",
    "sphere.partition.follow_up": "后续发展",
    "sphere.partition.follow_up.desc": "在本文方法之上的扩展、分析与改进工作。",
    "sphere.partition.application": "跨域应用",
    "sphere.partition.application.desc": "把本文思想迁移到其他任务与领域的工作。",
    "sphere.partition.frontier": "最新前沿",
    "sphere.partition.frontier.desc": "本文之后发表的最新相关研究。",
    "sphere.partition.survey": "相关综述",
    "sphere.partition.survey.desc": "梳理该领域全貌的入门性综述。",
    "sphere.partition.library": "我的文献库",
    "sphere.partition.library.desc": "来自你自己知识库的相关论文。",
    "sphere.themes.intro": "基于引文结构相似度（文献耦合）划分的社区。",
    "sphere.timeline.intro": "该研究脉络逐年的演进过程。",
    "sphere.cmp.intro": "{n} 篇论文横向对比，点击展开完整维度。",
    "sphere.cmp.paper": "论文",
    "sphere.cmp.problem": "问题",
    "sphere.cmp.method": "方法",
    "sphere.cmp.dataset": "数据集",
    "sphere.cmp.metric": "指标",
    "sphere.cmp.strength": "优势",
    "sphere.cmp.weakness": "不足",
    "sphere.hubs.intro": "引文网络中结构位置最核心的论文。",
    "sphere.hubs.pagerank": "相似图上的 PageRank 值",
    "sphere.gaps.evidence": "证据来源",
    "sphere.path.fast": "速读",
    "sphere.path.deep": "精读",
    "sphere.path.frontier": "前沿",
    "sphere.path.steps": "{n} 篇",
    "sphere.library.intro": "以下论文来自你自己的知识库，与本文语义相关。",
  },
};

interface LanguageContextValue {
  locale: Locale;
  setLocale: (l: Locale) => void;
  t: (key: string, vars?: Record<string, string | number>) => string;
}

const LanguageContext = createContext<LanguageContextValue>({
  locale: "en",
  setLocale: () => {},
  t: (key) => key,
});

const STORAGE_KEY = "scholar-locale";

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("en");

  // Load from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "zh" || saved === "en") {
      setLocaleState(saved);
    }
  }, []);

  const setLocale = useCallback((l: Locale) => {
    setLocaleState(l);
    localStorage.setItem(STORAGE_KEY, l);
  }, []);

  const t = useCallback(
    (key: string, vars?: Record<string, string | number>) => {
      let text = translations[locale][key] ?? translations.en[key] ?? key;
      if (vars) {
        for (const [k, v] of Object.entries(vars)) {
          text = text.replace(`{${k}}`, String(v));
        }
      }
      return text;
    },
    [locale],
  );

  return (
    <LanguageContext.Provider value={{ locale, setLocale, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useTranslation() {
  return useContext(LanguageContext);
}

export function LanguageToggle() {
  const { locale, setLocale } = useTranslation();

  return (
    <button
      onClick={() => setLocale(locale === "en" ? "zh" : "en")}
      className="rounded-full border border-border px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:border-foreground/25 hover:text-foreground"
      title={locale === "en" ? "切换到中文" : "Switch to English"}
    >
      {locale === "en" ? "中文" : "EN"}
    </button>
  );
}
