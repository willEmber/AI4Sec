"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { uploadPaper, createRun } from "@/lib/api";
import { useTranslation } from "@/lib/i18n";
import type { ReadingMode } from "@/lib/types";

const MODE_KEYS: { value: ReadingMode; labelKey: string; descKey: string }[] = [
  { value: "snap", labelKey: "upload.mode.snap.label", descKey: "upload.mode.snap.desc" },
  { value: "lens", labelKey: "upload.mode.lens.label", descKey: "upload.mode.lens.desc" },
  { value: "sphere", labelKey: "upload.mode.sphere.label", descKey: "upload.mode.sphere.desc" },
];

type OutputLanguage = "en" | "zh";

export default function UploadPage() {
  const router = useRouter();
  const { t, locale } = useTranslation();
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<ReadingMode>("snap");
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
    setUploading(true);
    setError(null);

    try {
      const uploadRes = await uploadPaper(file);
      const runRes = await createRun({
        paper_id: uploadRes.paper_id,
        mode,
        llm_model: llmModel,
        language: outputLanguage,
      });
      router.push(`/paper/${uploadRes.paper_id}/run/${runRes.run_id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : t("upload.fail"));
      setUploading(false);
    }
  }, [file, mode, llmModel, outputLanguage, router, t]);

  return (
    <div className="max-w-2xl mx-auto px-6 py-10">
      <h1 className="text-2xl font-bold mb-6">{t("upload.title")}</h1>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-colors mb-6 ${
          dragOver ? "border-[var(--primary)] bg-[var(--accent)]" : "border-[var(--border)]"
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
        {file ? (
          <div>
            <p className="font-medium">{file.name}</p>
            <p className="text-sm text-[var(--muted-foreground)]">
              {(file.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        ) : (
          <div>
            <p className="font-medium mb-1">{t("upload.drop")}</p>
            <p className="text-sm text-[var(--muted-foreground)]">{t("upload.supports")}</p>
          </div>
        )}
      </div>

      {/* Mode selection */}
      <div className="mb-6">
        <label className="block font-medium mb-3">{t("upload.mode_label")}</label>
        <div className="grid gap-3 sm:grid-cols-3">
          {MODE_KEYS.map((m) => (
            <button
              key={m.value}
              onClick={() => setMode(m.value)}
              className={`border rounded-lg p-4 text-left transition-colors ${
                mode === m.value
                  ? "border-[var(--primary)] bg-[var(--accent)]"
                  : "border-[var(--border)] hover:border-[var(--muted-foreground)]"
              }`}
            >
              <p className="font-medium text-sm">{t(m.labelKey)}</p>
              <p className="text-xs text-[var(--muted-foreground)] mt-1">{t(m.descKey)}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Output Language selection */}
      <div className="mb-6">
        <label className="block font-medium mb-3">{t("upload.language_label")}</label>
        <div className="grid gap-3 sm:grid-cols-2">
          <button
            onClick={() => setOutputLanguage("en")}
            className={`border rounded-lg p-4 text-left transition-colors ${
              outputLanguage === "en"
                ? "border-[var(--primary)] bg-[var(--accent)]"
                : "border-[var(--border)] hover:border-[var(--muted-foreground)]"
            }`}
          >
            <p className="font-medium text-sm">English</p>
          </button>
          <button
            onClick={() => setOutputLanguage("zh")}
            className={`border rounded-lg p-4 text-left transition-colors ${
              outputLanguage === "zh"
                ? "border-[var(--primary)] bg-[var(--accent)]"
                : "border-[var(--border)] hover:border-[var(--muted-foreground)]"
            }`}
          >
            <p className="font-medium text-sm">中文</p>
          </button>
        </div>
        <p className="text-xs text-[var(--muted-foreground)] mt-2">{t("upload.language_note")}</p>
      </div>

      {/* Model selection */}
      <div className="mb-6">
        <label className="block font-medium mb-2">{t("upload.model_label")}</label>
        <input
          type="text"
          value={llmModel}
          onChange={(e) => setLlmModel(e.target.value)}
          placeholder={t("upload.model_placeholder")}
          className="w-full border border-[var(--border)] rounded-lg px-4 py-2 bg-transparent text-sm"
        />
      </div>

      {/* Error */}
      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-50 dark:bg-red-950 text-[var(--destructive)] text-sm">
          {error}
        </div>
      )}

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={!file || uploading}
        className="w-full bg-[var(--primary)] text-[var(--primary-foreground)] py-3 rounded-lg font-medium disabled:opacity-50 hover:opacity-90 transition-opacity"
      >
        {uploading ? t("upload.submitting") : t("upload.submit")}
      </button>
    </div>
  );
}
