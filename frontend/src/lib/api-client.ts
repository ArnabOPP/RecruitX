"use client";

import {
  AnswerResponse,
  ApiError,
  CodeSubmissionResponse,
  CreateSessionResponse,
  EnrollResponse,
  ProctoringSnapshotResponse,
  SessionReport,
} from "./types";

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let body = null;
    try {
      body = await response.json();
    } catch {
      // non-JSON error body — ApiError falls back to a generic message
    }
    throw new ApiError(response.status, body);
  }
  return response.json() as Promise<T>;
}

export async function enrollCandidate(candidateId: string, faces: Blob[]): Promise<EnrollResponse> {
  const form = new FormData();
  form.set("candidate_id", candidateId);
  faces.forEach((blob, i) => form.append("files", blob, `enroll-${i}.jpg`));
  const response = await fetch("/api/biometric/enroll", { method: "POST", body: form });
  return handleResponse(response);
}

export interface CreateSessionOptions {
  resumeFile: File;
  candidateId?: string;
  faceFiles?: Blob[];
  targetCompany?: string;
  personalQuestionCount?: number;
  hrQuestionCount?: number;
  enableFollowups?: boolean;
}

export async function createSession(options: CreateSessionOptions): Promise<CreateSessionResponse> {
  const form = new FormData();
  form.set("file", options.resumeFile);
  if (options.candidateId) form.set("candidate_id", options.candidateId);
  options.faceFiles?.forEach((blob, i) => form.append("face_files", blob, `verify-${i}.jpg`));
  if (options.targetCompany) form.set("target_company", options.targetCompany);
  if (options.personalQuestionCount !== undefined) form.set("personal_question_count", String(options.personalQuestionCount));
  if (options.hrQuestionCount !== undefined) form.set("hr_question_count", String(options.hrQuestionCount));
  if (options.enableFollowups !== undefined) form.set("enable_followups", String(options.enableFollowups));

  const response = await fetch("/api/sessions", { method: "POST", body: form });
  return handleResponse(response);
}

export async function submitTextAnswer(sessionId: string, answerText: string): Promise<AnswerResponse> {
  const response = await fetch(`/api/sessions/${sessionId}/answer`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ answer_text: answerText }),
  });
  return handleResponse(response);
}

export async function submitAudioAnswer(sessionId: string, audio: Blob): Promise<AnswerResponse> {
  const form = new FormData();
  form.set("file", audio, "answer.webm");
  const response = await fetch(`/api/sessions/${sessionId}/answer/audio`, { method: "POST", body: form });
  return handleResponse(response);
}

export async function submitProctoringSnapshot(sessionId: string, frame: Blob): Promise<ProctoringSnapshotResponse> {
  const form = new FormData();
  form.set("file", frame, "frame.jpg");
  const response = await fetch(`/api/sessions/${sessionId}/proctoring/snapshot`, { method: "POST", body: form });
  return handleResponse(response);
}

export interface CodeSubmissionOptions {
  language: string;
  sourceCode: string;
  testCases: Array<{ input?: string; expected_output: string }>;
  expectedComplexity?: string;
}

export async function submitCode(sessionId: string, options: CodeSubmissionOptions): Promise<CodeSubmissionResponse> {
  const response = await fetch(`/api/sessions/${sessionId}/code`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      language: options.language,
      source_code: options.sourceCode,
      test_cases: options.testCases,
      expected_complexity: options.expectedComplexity ?? null,
    }),
  });
  return handleResponse(response);
}

export async function fetchReport(sessionId: string): Promise<SessionReport> {
  const response = await fetch(`/api/sessions/${sessionId}/report`, { cache: "no-store" });
  return handleResponse(response);
}
