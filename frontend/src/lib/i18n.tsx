"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

export type Locale = "en" | "zh";

const translations: Record<Locale, Record<string, string>> = {
  en: {
    // Nav
    "nav.brand": "Scholar",
    "nav.upload": "Upload & Analyze",

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
    "upload.model_label": "LLM Model (optional)",
    "upload.model_placeholder": "Leave empty for default model",
    "upload.language_label": "Output Language",
    "upload.language_note": "Research workflow uses English internally; only the final output is translated.",
    "upload.submit": "Start Analysis",
    "upload.submitting": "Uploading & starting analysis...",
    "upload.fail": "Upload failed",

    // Run page
    "run.mode": "Mode",
    "run.status.running": "Running...",
    "run.status.complete": "Complete",
    "run.status.failed_label": "Analysis Failed",
    "run.status.unknown": "An unknown error occurred",
    "run.starting": "Starting analysis...",
    "run.export_md": "Export .md",

    // Steps
    "step.ingest_pdf": "Verifying PDF",
    "step.mineru_parse": "Parsing with MinerU",
    "step.build_paper_ir": "Building document structure",
    "step.enrich_metadata": "Looking up publication rank",
    "step.run_snap": "Generating Insight Snap",
    "step.run_lens": "Generating Logic Lens",
    "step.run_sphere": "Generating Research Sphere",
    "step.translate_output": "Translating output",
    "step.persist_output": "Saving results",
    "step.sphere_init_from_pdf": "Extracting references from PDF",
    "step.extract_core_metadata": "Extracting core metadata",
    "step.resolve_canonical_ids": "Resolving paper identifiers",
    "step.expand_graph_candidates": "Expanding citation graph",
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
  },
  zh: {
    // Nav
    "nav.brand": "Scholar",
    "nav.upload": "上传与分析",

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
    "upload.model_label": "LLM 模型（可选）",
    "upload.model_placeholder": "留空使用默认模型",
    "upload.language_label": "输出语言",
    "upload.language_note": "研究流程内部使用英文；仅最终输出会被翻译。",
    "upload.submit": "开始分析",
    "upload.submitting": "正在上传并启动分析...",
    "upload.fail": "上传失败",

    // Run page
    "run.mode": "模式",
    "run.status.running": "运行中...",
    "run.status.complete": "已完成",
    "run.status.failed_label": "分析失败",
    "run.status.unknown": "发生未知错误",
    "run.starting": "正在启动分析...",
    "run.export_md": "导出 .md",

    // Steps
    "step.ingest_pdf": "验证PDF文件",
    "step.mineru_parse": "MinerU解析中",
    "step.build_paper_ir": "构建文档结构",
    "step.enrich_metadata": "查询期刊等级",
    "step.run_snap": "生成快速洞察",
    "step.run_lens": "生成逻辑透镜",
    "step.run_sphere": "生成研究全景",
    "step.translate_output": "翻译输出内容",
    "step.persist_output": "保存结果",
    "step.sphere_init_from_pdf": "从PDF提取参考文献",
    "step.extract_core_metadata": "提取核心元数据",
    "step.resolve_canonical_ids": "解析论文标识符",
    "step.expand_graph_candidates": "扩展引用图谱",
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
      className="px-2 py-1 text-sm border border-[var(--border)] rounded-md hover:bg-[var(--accent)] transition-colors"
      title={locale === "en" ? "切换到中文" : "Switch to English"}
    >
      {locale === "en" ? "中文" : "EN"}
    </button>
  );
}
