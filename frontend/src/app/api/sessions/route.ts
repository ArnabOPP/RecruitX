import { NextRequest } from "next/server";
import { ORCHESTRATOR_BASE_URL, orchestratorHeaders, relay } from "@/lib/backend";

/** Proxies résumé upload (+ optional face_files/candidate_id for identity
 * verification) straight through to the orchestrator. candidate_id and
 * the round-size params are query params on the orchestrator's own
 * endpoint (mixed with multipart file fields), not form fields — see
 * services/orchestrator/app/main.py's create_session_endpoint signature. */
export async function POST(request: NextRequest) {
  const incomingForm = await request.formData();

  const forwardForm = new FormData();
  const file = incomingForm.get("file");
  if (file) forwardForm.set("file", file);
  for (const faceFile of incomingForm.getAll("face_files")) {
    forwardForm.append("face_files", faceFile);
  }

  const params = new URLSearchParams();
  for (const key of ["candidate_id", "target_company", "personal_question_count", "hr_question_count", "enable_followups"]) {
    const value = incomingForm.get(key);
    if (typeof value === "string" && value.length > 0) params.set(key, value);
  }

  const backendResponse = await fetch(`${ORCHESTRATOR_BASE_URL}/api/v1/sessions?${params.toString()}`, {
    method: "POST",
    body: forwardForm,
    headers: orchestratorHeaders(),
  });
  return relay(backendResponse);
}
