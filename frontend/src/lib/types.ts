/** Data contracts mirroring the orchestrator's/biometric-auth's Pydantic
 * schemas (services/orchestrator/app/schemas.py,
 * services/biometric-auth/app/schemas.py). Kept as plain interfaces, not
 * re-validated client-side — the backend is the source of truth for
 * shape; this is just enough typing to catch typos while wiring the UI. */

export interface QuestionOut {
  text: string;
  category: string | null;
  grounding: Record<string, unknown> | null;
  difficulty: string | null;
  round: string;
  stage: string | null;
}

export interface CreateSessionResponse {
  session_id: string;
  status: string;
  round: string;
  question: QuestionOut | null;
}

export interface AnswerResponse {
  score: {
    overall_score: number;
    rubric_source: string;
    criteria_scores: unknown[];
    explanation: string;
  };
  round: string;
  status: string;
  next_question: QuestionOut | null;
}

export interface CodeSubmissionResponse {
  result: {
    overall_score: number;
    correctness: { passed: number; total: number; pass_rate: number };
    [key: string]: unknown;
  };
  status: string;
}

export interface ProctoringEvent {
  type: string;
  timestamp: number;
  severity: number;
}

export interface ProctoringSummary {
  session_id: string;
  frames_processed: number;
  integrity_score: number;
  event_counts: Record<string, number>;
  events: ProctoringEvent[];
}

export interface SessionReport {
  session_id: string;
  status: string;
  round: string;
  history: Array<{
    round: string;
    stage: string;
    question?: string;
    grounding?: unknown;
    answer?: string;
    score?: AnswerResponse["score"];
    language?: string;
    result?: CodeSubmissionResponse["result"];
  }>;
  overall_average_score: number | null;
  proctoring_summary: ProctoringSummary | null;
}

export interface ProctoringSnapshotResponse {
  session_id: string;
  faces_detected: number;
  head_pose_deviation_degrees: number | null;
  gaze_offset: number | null;
  flagged_this_frame: string[];
  events_recorded: string[];
  integrity_score: number;
  frames_processed: number;
}

export interface EnrollResponse {
  candidate_id: string;
  enrolled: boolean;
  images_used: number;
  model_used: string;
}

export interface ApiErrorBody {
  error: string;
  detail: string;
  request_id?: string | null;
}

export class ApiError extends Error {
  readonly status: number;
  readonly body: ApiErrorBody | null;

  constructor(status: number, body: ApiErrorBody | null) {
    super(body?.detail || `Request failed with status ${status}`);
    this.status = status;
    this.body = body;
  }
}
