"use client";

import { useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { listSessions } from "@/lib/agent";
import { getMe } from "@/lib/auth";
import { useTranslation } from "@/lib/i18n";
import {
  IconSnap,
  IconLens,
  IconSphere,
  IconSparkles,
  IconArrowRight,
} from "@/components/icons";
import type { ComponentType } from "react";

const MODES: {
  key: string;
  titleKey: string;
  descKey: string;
  Icon: ComponentType<{ className?: string }>;
}[] = [
  { key: "snap", titleKey: "home.mode.snap.title", descKey: "home.mode.snap.desc", Icon: IconSnap },
  { key: "lens", titleKey: "home.mode.lens.title", descKey: "home.mode.lens.desc", Icon: IconLens },
  { key: "sphere", titleKey: "home.mode.sphere.title", descKey: "home.mode.sphere.desc", Icon: IconSphere },
];

const CAPABILITIES = ["home.cap.multiturn", "home.cap.tools", "home.cap.context", "home.cap.memory"];

export default function Home() {
  const { t } = useTranslation();
  const router = useRouter();

  // Returning readers, logged-in accounts and local single-user setups go
  // straight into the conversation workspace instead of clicking through a
  // marketing splash every time.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const me = await getMe();
      if (cancelled || !me) return;
      if (me.mode === "single_user" || me.authenticated) {
        router.replace("/chat");
        return;
      }
      if (!me.principal_id) return;
      const sess = await listSessions().catch(() => ({ sessions: [] }));
      if (!cancelled && sess.sessions.length > 0) {
        router.replace("/chat");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  return (
    <div className="mx-auto max-w-5xl px-6 pb-24 pt-16 sm:pt-24">
      {/* Hero */}
      <section className="animate-fade-in-up text-center">
        <span className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3.5 py-1.5 text-xs font-medium text-muted-foreground soft-shadow">
          <span className="h-1.5 w-1.5 rounded-full bg-primary" />
          {t("home.eyebrow")}
        </span>

        <h1 className="font-display mx-auto mt-7 max-w-3xl text-balance text-4xl font-semibold leading-[1.1] tracking-tight sm:text-6xl">
          {t("home.title")}
        </h1>

        <p className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-muted-foreground sm:text-lg">
          {t("home.subtitle")}
        </p>

        <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/chat"
            className="group inline-flex items-center gap-2 rounded-xl bg-primary px-6 py-3.5 font-medium text-primary-foreground transition-colors hover:bg-primary-hover"
          >
            <IconSparkles className="text-lg" />
            {t("home.cta")}
            <IconArrowRight className="text-lg transition-transform group-hover:translate-x-0.5" />
          </Link>
        </div>

        <ul className="mx-auto mt-8 flex max-w-2xl flex-wrap justify-center gap-x-5 gap-y-2 text-xs text-muted-foreground">
          {CAPABILITIES.map((key) => (
            <li key={key} className="inline-flex items-center gap-1.5">
              <span className="h-1 w-1 rounded-full bg-primary/60" />
              {t(key)}
            </li>
          ))}
        </ul>
      </section>

      {/* Modes */}
      <section className="mt-20">
        <div className="mb-8 text-center">
          <h2 className="font-display text-2xl font-semibold tracking-tight sm:text-3xl">
            {t("home.modes_heading")}
          </h2>
          <p className="mt-2 text-sm text-muted-foreground">{t("home.modes_sub")}</p>
        </div>

        <div className="grid gap-5 sm:grid-cols-3">
          {MODES.map(({ key, titleKey, descKey, Icon }) => (
            <Link
              key={key}
              href={`/chat?mode=${key}`}
              className="lift group rounded-2xl border border-border bg-card p-6 soft-shadow"
            >
              <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-accent text-xl text-primary">
                <Icon />
              </span>
              <h3 className="mt-5 text-lg font-semibold">{t(titleKey)}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                {t(descKey)}
              </p>
              <span className="mt-4 inline-flex items-center gap-1 text-xs font-medium text-primary">
                {t("home.mode.start")}
                <IconArrowRight className="transition-transform group-hover:translate-x-0.5" />
              </span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
