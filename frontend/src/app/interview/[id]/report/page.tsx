"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { fetchReport } from "@/lib/api-client";
import { ApiError, SessionReport } from "@/lib/types";

function scorePercent(score: number | undefined | null): string {
  if (score === undefined || score === null) return "—";
  return `${Math.round(score * 100)}%`;
}

export default function ReportPage() {
  const params = useParams<{ id: string }>();
  const sessionId = params.id;

  const [report, setReport] = useState<SessionReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetchReport(sessionId)
      .then((data) => {
        if (!cancelled) setReport(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Could not load the report.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  if (loading) {
    return (
      <main className="mx-auto flex w-full max-w-2xl flex-1 items-center justify-center px-6">
        <p className="text-sm text-neutral-500">Loading report…</p>
      </main>
    );
  }

  if (error || !report) {
    return (
      <main className="mx-auto flex w-full max-w-2xl flex-1 items-center justify-center px-6">
        <p className="text-sm text-red-600">{error ?? "Report not found."}</p>
      </main>
    );
  }

  const proctoring = report.proctoring_summary;

  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-8 px-6 py-12">
      <div>
        <h1 className="text-2xl font-semibold">Interview report</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Session {report.session_id} · Status: {report.status}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-700">
          <p className="text-xs uppercase tracking-wide text-neutral-400">Overall score</p>
          <p className="mt-1 text-2xl font-semibold">{scorePercent(report.overall_average_score)}</p>
        </div>
        <div className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-700">
          <p className="text-xs uppercase tracking-wide text-neutral-400">Integrity score</p>
          <p className="mt-1 text-2xl font-semibold">
            {proctoring ? Math.round(proctoring.integrity_score) : "No proctoring data"}
          </p>
        </div>
      </div>

      {proctoring && (
        <div className="rounded-lg border border-neutral-200 p-4 text-sm dark:border-neutral-700">
          <p className="font-medium">Proctoring summary</p>
          <p className="mt-1 text-neutral-500">{proctoring.frames_processed} frame(s) checked.</p>
          {Object.keys(proctoring.event_counts).length > 0 ? (
            <ul className="mt-2 list-inside list-disc text-neutral-600 dark:text-neutral-300">
              {Object.entries(proctoring.event_counts).map(([type, count]) => (
                <li key={type}>
                  {type.replaceAll("_", " ")}: {count}
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-neutral-500">No integrity events recorded.</p>
          )}
        </div>
      )}

      <div className="flex flex-col gap-4">
        <h2 className="text-lg font-medium">Transcript</h2>
        {report.history.map((entry, i) => (
          <div key={i} className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-700">
            <span className="text-xs uppercase tracking-wide text-neutral-400">
              {entry.round} · {entry.stage}
            </span>
            {entry.question && <p className="mt-1 text-sm font-medium">{entry.question}</p>}
            {entry.answer && <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-300">{entry.answer}</p>}
            {entry.score && (
              <p className="mt-2 text-sm text-neutral-500">Score: {scorePercent(entry.score.overall_score)}</p>
            )}
            {entry.result && (
              <p className="mt-2 text-sm text-neutral-500">
                Code: {entry.result.correctness.passed}/{entry.result.correctness.total} tests passed
              </p>
            )}
          </div>
        ))}
      </div>
    </main>
  );
}
