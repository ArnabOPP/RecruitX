"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { WebcamCapture } from "@/components/WebcamCapture";
import { createSession, enrollCandidate } from "@/lib/api-client";
import { saveSessionState } from "@/lib/session-cache";
import { ApiError } from "@/lib/types";

type Step = "details" | "enroll" | "verify" | "creating";

function randomCandidateId(): string {
  return `cand-${Math.random().toString(36).slice(2, 10)}`;
}

export default function HomePage() {
  const router = useRouter();

  const [step, setStep] = useState<Step>("details");
  const [candidateId, setCandidateId] = useState(randomCandidateId());
  const [targetCompany, setTargetCompany] = useState("");
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [useFaceVerification, setUseFaceVerification] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [enrolling, setEnrolling] = useState(false);

  const handleContinueFromDetails = () => {
    setError(null);
    if (!resumeFile) {
      setError("Please choose a résumé file first.");
      return;
    }
    if (!candidateId.trim()) {
      setError("A candidate ID is required.");
      return;
    }
    if (useFaceVerification) {
      setStep("enroll");
    } else {
      void handleCreateSession(null);
    }
  };

  const handleEnrollComplete = async (frames: Blob[]) => {
    setError(null);
    setEnrolling(true);
    try {
      await enrollCandidate(candidateId, frames);
      setStep("verify");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Enrollment failed.");
      setStep("details");
    } finally {
      setEnrolling(false);
    }
  };

  const handleVerifyComplete = async (frames: Blob[]) => {
    await handleCreateSession(frames);
  };

  const handleCreateSession = async (faceFiles: Blob[] | null) => {
    setError(null);
    setStep("creating");
    try {
      if (!resumeFile) throw new Error("Missing résumé file.");
      const session = await createSession({
        resumeFile,
        candidateId,
        faceFiles: faceFiles ?? undefined,
        targetCompany: targetCompany || undefined,
      });
      saveSessionState({
        sessionId: session.session_id,
        status: session.status,
        round: session.round,
        question: session.question,
      });
      router.push(`/interview/${session.session_id}`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not start the interview session.");
      setStep("details");
    }
  };

  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col justify-center gap-8 px-6 py-16">
      <div>
        <h1 className="text-2xl font-semibold">Recruitix</h1>
        <p className="mt-1 text-sm text-neutral-500">AI-driven interview practice, start to finish.</p>
      </div>

      {error && (
        <p className="rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
          {error}
        </p>
      )}

      {step === "details" && (
        <div className="flex flex-col gap-5">
          <label className="flex flex-col gap-1.5 text-sm">
            <span className="font-medium">Candidate ID</span>
            <input
              value={candidateId}
              onChange={(e) => setCandidateId(e.target.value)}
              className="rounded-md border border-neutral-300 px-3 py-2 dark:border-neutral-600 dark:bg-neutral-900"
            />
          </label>

          <label className="flex flex-col gap-1.5 text-sm">
            <span className="font-medium">Target company (optional)</span>
            <input
              value={targetCompany}
              onChange={(e) => setTargetCompany(e.target.value)}
              className="rounded-md border border-neutral-300 px-3 py-2 dark:border-neutral-600 dark:bg-neutral-900"
            />
          </label>

          <label className="flex flex-col gap-1.5 text-sm">
            <span className="font-medium">Résumé (PDF, DOCX, or TXT)</span>
            <input
              type="file"
              accept=".pdf,.docx,.txt"
              onChange={(e) => setResumeFile(e.target.files?.[0] ?? null)}
              className="text-sm"
            />
          </label>

          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={useFaceVerification}
              onChange={(e) => setUseFaceVerification(e.target.checked)}
            />
            Verify my identity with face recognition before starting
          </label>

          <button
            type="button"
            onClick={handleContinueFromDetails}
            className="rounded-md bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white"
          >
            Continue
          </button>
        </div>
      )}

      {step === "enroll" && (
        <div className="flex flex-col items-center gap-4">
          <h2 className="text-lg font-medium">Enroll your face</h2>
          <p className="max-w-sm text-center text-sm text-neutral-500">
            Capture 3 clear photos of your face. These are used only to verify it&apos;s really you at the start of
            the interview.
          </p>
          <WebcamCapture targetCount={3} label="Enrollment" onComplete={handleEnrollComplete} />
          {enrolling && <p className="text-sm text-neutral-500">Enrolling…</p>}
        </div>
      )}

      {step === "verify" && (
        <div className="flex flex-col items-center gap-4">
          <h2 className="text-lg font-medium">Verify it&apos;s you</h2>
          <p className="max-w-sm text-center text-sm text-neutral-500">
            One more photo to confirm your identity matches your enrollment.
          </p>
          <WebcamCapture targetCount={1} label="Verification" onComplete={handleVerifyComplete} />
        </div>
      )}

      {step === "creating" && <p className="text-center text-sm text-neutral-500">Starting your interview…</p>}
    </main>
  );
}
