"use client";

import { QuestionOut } from "./types";

/** The orchestrator has no "get current session state" endpoint — only
 * /answer, /code, and /report responses carry status/round/question, and
 * /report doesn't include the still-pending question. This sessionStorage
 * cache is what lets the interview page survive a refresh with its
 * current question intact; it's a UI convenience, not a source of truth
 * (the backend's own session_data in Redis is that). */
export interface CachedSessionState {
  sessionId: string;
  status: string;
  round: string;
  question: QuestionOut | null;
}

const keyFor = (sessionId: string) => `recruitix:session:${sessionId}`;

export function saveSessionState(state: CachedSessionState): void {
  try {
    sessionStorage.setItem(keyFor(state.sessionId), JSON.stringify(state));
  } catch {
    // sessionStorage unavailable (private browsing, SSR) — non-fatal
  }
}

export function loadSessionState(sessionId: string): CachedSessionState | null {
  try {
    const raw = sessionStorage.getItem(keyFor(sessionId));
    return raw ? (JSON.parse(raw) as CachedSessionState) : null;
  } catch {
    return null;
  }
}
