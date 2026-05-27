"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { uploadPaper, createRun } from "@/lib/api";
import { useTranslation } from "@/lib/i18n";
import type { ReadingMode } from "@/lib/types";
import {
  IconSnap,
  IconLens,
  IconSphere,
  IconSparkles,
  IconUpload,
  IconCheck,
} from "@/components/icons";
import type { ComponentType } from "react";

const MODE_KEYS: {
  value: ReadingMode;
  labelKey: string;
  descKey: string;
  Icon: ComponentType<{ className?: string }>;
}[] = [
  { value: "snap", labelKey: "upload.mode.snap.label", descKey: "upload.mode.snap.desc", Icon: IconSnap },
  { value: "lens", labelKey: "upload.mode.lens.label", descKey: "upload.mode.lens.desc", Icon: IconLens },
  { value: "sphere", labelKey: "upload.mode.sphere.label", descKey: "upload.mode.sphere.desc", Icon: IconSphere },
  { value: "auto", labelKey: "upload.mode.auto.label", descKey: "upload.mode.auto.desc", Icon: IconSparkles },
];

type OutputLanguage = "en" | "zh";

export default function UploadPage() {
  const router = useRouter();
  const { t, locale } = useTranslation();
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<ReadingMode>("snap");
  const [question, setQuestion] = useState("");
  const [llmModel, setLlmModel] = useState("");
  const [outputLanguage, setOutputLanguage] = useState<OutputLanguage>(locale);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped?.type === "application/pdf") {
      setFile(dropped);
      setError(null);
    } else {
      setError(t("upload.drop_error"));
    }
  }, [t]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setError(null);
    }
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!file) return;
    if (mode === "auto" && !question.trim()) {
      setError(t("upload.question_required"));
      return;
    }
    setUploading(true);
    setError(null);

    try {
      const uploadRes = await uploadPaper(file);
      const runRes = await createRun({
        paper_id: uploadRes.paper_id,
        mode,
        llm_model: llmModel,
        language: outputLanguage,
        question: mode === "auto" ? question.trim() : "",
      });
      router.push(`/paper/${uploadRes.paper_id}/run/${runRes.run_id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : t("upload.fail"));
      setUploading(false);
    }
  }, [file, mode, question, llmModel, outputLanguage, router, t]);

  const langBtn = (value: OutputLanguage, label: string) => (
    <button
      onClick={() => setOutputLanguage(value)}
      className={`rounded-xl border p-4 text-center text-sm font-medium transition-colors ${
        outputLanguage === value
          ? "border-primary bg-accent text-accent-foreground"
          : "border-border hover:border-foreground/20"
      }`}
    >
      {label}
    </button>
  );

  return (
    <div className="mx-auto max-w-2xl px-6 py-12">
      <header className="mb-8">
        <h1 className="font-display text-3xl font-semibold tracking-tight">
          {t("upload.title")}
        </h1>
      </header>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`mb-8 cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition-colors ${
          dragOver
            ? "border-primary bg-accent"
            : file
              ? "border-primary/40 bg-card"
              : "border-border bg-card hover:border-foreground/20"
        }`}
        onClick={() => document.getElementById("file-input")?.click()}
      >
        <input
          id="file-input"
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          className="hidden"
        />
        <span
          className={`mx-auto flex h-12 w-12 items-center justify-center rounded-2xl text-2xl ${
            file ? "bg-accent text-primary" : "bg-muted text-muted-foreground"
          }`}
        >
          {file ? <IconCheck /> : <IconUpload />}
        </span>
        {file ? (
          <div className="mt-4">
            <p className="font-medium">{file.name}</p>
            <p className="text-sm text-muted-foreground">
              {(file.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        ) : (
          <div className="mt-4">
            <p className="font-medium">{t("upload.drop")}</p>
            <p className="mt-1 text-sm text-muted-foreground">{t("upload.supports")}</p>
          </div>
        )}
      </div>

      {/* Mode selection */}
      <div className="mb-8">
        <label className="mb-3 block text-sm font-semibold">{t("upload.mode_label")}</label>
        <div className="grid gap-3 sm:grid-cols-2">
          {MODE_KEYS.map((m) => {
            const selected = mode === m.value;
            return (
              <button
                key={m.value}
                onClick={() => setMode(m.value)}
                className={`relative rounded-xl border p-4 text-left transition-colors ${
                  selected
                    ? "border-primary bg-accent"
                    : "border-border hover:border-foreground/20"
                }`}
              >
                {selected && (
                  <span className="absolute right-3 top-3 flex h-5 w-5 items-center justify-center rounded-full bg-primary text-xs text-primary-foreground">
                    <IconCheck />
                  </span>
                )}
                <span
                  className={`flex h-9 w-9 items-center justify-center rounded-lg text-lg ${
                    selected ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"
                  }`}
                >
                  <m.Icon />
                </span>
                <p className="mt-3 text-sm font-medium">{t(m.labelKey)}</p>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{t(m.descKey)}</p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Smart Q&A question input (only visible when mode === "auto") */}
      {mode === "auto" && (
        <div className="mb-8">
          <label className="mb-2 block text-sm font-semibold">{t("upload.question_label")}</label>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={t("upload.question_placeholder")}
            rows={3}
            maxLength={2000}
            className="min-h-[88px] w-full resize-y rounded-xl border border-border bg-card px-4 py-3 text-sm transition-colors placeholder:text-muted-foreground/70 focus:border-primary focus:outline-none"
          />
          <p className="mt-2 text-xs text-muted-foreground">{t("upload.question_hint")}</p>
        </div>
      )}

      {/* Output Language selection */}
      <div className="mb-8">
        <label className="mb-3 block text-sm font-semibold">{t("upload.language_label")}</label>
        <div className="grid gap-3 sm:grid-cols-2">
          {langBtn("en", "English")}
          {langBtn("zh", "中文")}
        </div>
        <p className="mt-2 text-xs text-muted-foreground">{t("upload.language_note")}</p>
      </div>

      {/* Model selection */}
      <div className="mb-8">
        <label className="mb-2 block text-sm font-semibold">{t("upload.model_label")}</label>
        <input
          type="text"
          value={llmModel}
          onChange={(e) => setLlmModel(e.target.value)}
          placeholder={t("upload.model_placeholder")}
          className="w-full rounded-xl border border-border bg-card px-4 py-3 text-sm transition-colors placeholder:text-muted-foreground/70 focus:border-primary focus:outline-none"
        />
      </div>

      {/* Error */}
      {error && (
        <div className="mb-5 rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={!file || uploading}
        className="flex w-full items-center justify-center gap-2 rounded-xl bg-primary py-3.5 font-medium text-primary-foreground transition-colors hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-primary"
      >
        {uploading && (
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground/40 border-t-primary-foreground" />
        )}
        {uploading ? t("upload.submitting") : t("upload.submit")}
      </button>
    </div>
  );
}
