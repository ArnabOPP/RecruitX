"use client";

import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { AnswerInput } from "@/components/AnswerInput";
import { CodingRound } from "@/components/CodingRound";
import { ProctoringMonitor } from "@/components/ProctoringMonitor";
import { submitAudioAnswer, submitTextAnswer } from "@/lib/api-client";
import { loadSessionState, saveSessionState } from "@/lib/session-cache";
import { ApiError, CodeSubmissionResponse, QuestionOut } from "@/lib/types";

const SNAPSHOT_INTERVAL_MS = Number(process.env.NEXT_PUBLIC_PROCTORING_SNAPSHOT_INTERVAL_MS ?? 8000);

const ROUND_LABEL: Record<string, string> = {
  personal: "Personal round",
  hr: "HR round",
  coding: "Coding round",
};

export default function InterviewPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const sessionId = params.id;

  // Lazy initializers, not an effect: sessionStorage is a synchronous
  // read, so there's no async state to synchronize with — reading it
  // during render and seeding state directly avoids the extra
  // render-then-setState pass an effect would cause. loadSessionState
  // only needs to run once per mount, which useState's lazy-initializer
  // form guarantees regardless of how many state slots read from it.
  const [status, setStatus] = useState<string>(() => loadSessionState(sessionId)?.status ?? "unknown");
  const [round, setRound] = useState<string>(() => loadSessionState(sessionId)?.round ?? "personal");
  const [question, setQuestion] = useState<QuestionOut | null>(() => loadSessionState(sessionId)?.question ?? null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const persist = (next: { status: string; round: string; question: QuestionOut | null }) => {
    setStatus(next.status);
    setRound(next.round);
    setQuestion(next.question);
    saveSessionState({ sessionId, ...next });
  };

  const handleTextAnswer = async (text: string) => {
    setError(null);
    setSubmitting(true);
    try {
      const result = await submitTextAnswer(sessionId, text);
      persist({ status: result.status, round: result.round, question: result.next_question });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not submit your answer.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleAudioAnswer = async (audio: Blob) => {
    setError(null);
    setSubmitting(true);
    try {
      const result = await submitAudioAnswer(sessionId, audio);
      persist({ status: result.status, round: result.round, question: result.next_question });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not submit your recording.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleCodeCompleted = (result: CodeSubmissionResponse) => {
    persist({ status: result.status, round: "completed", question: null });
    router.push(`/interview/${sessionId}/report`);
  };

  useEffect(() => {
    if (status === "completed") router.push(`/interview/${sessionId}/report`);
  }, [status, sessionId, router]);

  const proctoringActive = status === "in_progress" || status === "awaiting_code";

  if (status === "unknown") {
    return (
      <main className="mx-auto flex w-full max-w-xl flex-1 flex-col items-center justify-center gap-4 px-6 text-center">
        <p className="text-sm text-neutral-500">
          This session&apos;s in-progress state isn&apos;t available (likely a page refresh). You can still view
          whatever&apos;s been recorded so far.
        </p>
        <a href={`/interview/${sessionId}/report`} className="text-sm font-medium text-indigo-600">
          View report →
        </a>
      </main>
    );
  }

  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-6 px-6 py-10">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">{ROUND_LABEL[round] ?? round}</h1>
        <ProctoringMonitor sessionId={sessionId} active={proctoringActive} intervalMs={SNAPSHOT_INTERVAL_MS} />
      </div>

      {error && (
        <p className="rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
          {error}
        </p>
      )}

      {status === "in_progress" && question && (
        <div className="flex flex-col gap-5">
          <div className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-700">
            {question.category && (
              <span className="text-xs uppercase tracking-wide text-neutral-400">{question.category}</span>
            )}
            <p className="mt-1 text-base">{question.text}</p>
          </div>
          <AnswerInput onSubmitText={handleTextAnswer} onSubmitAudio={handleAudioAnswer} disabled={submitting} />
          {submitting && <p className="text-sm text-neutral-500">Grading your answer…</p>}
        </div>
      )}

      {status === "awaiting_code" && (
        <div className="flex flex-col gap-4">
          <p className="text-sm text-neutral-500">Solve the problem below and submit it for grading.</p>
          <CodingRound sessionId={sessionId} onCompleted={handleCodeCompleted} />
        </div>
      )}
    </main>
  );
}
