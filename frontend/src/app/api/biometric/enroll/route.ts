import { NextRequest } from "next/server";
import { BIOMETRIC_AUTH_BASE_URL, biometricAuthHeaders, relay } from "@/lib/backend";

/** Enrollment is a one-time setup step, separate from any interview
 * session (see services/orchestrator/README.md) — proxied straight to
 * biometric-auth, not through the orchestrator. */
export async function POST(request: NextRequest) {
  const incomingForm = await request.formData();
  const candidateId = incomingForm.get("candidate_id");
  if (typeof candidateId !== "string" || !candidateId.trim()) {
    return Response.json({ error: "request_error", detail: "candidate_id is required." }, { status: 400 });
  }

  const forwardForm = new FormData();
  for (const file of incomingForm.getAll("files")) {
    forwardForm.append("files", file);
  }

  const params = new URLSearchParams({ candidate_id: candidateId });
  const backendResponse = await fetch(`${BIOMETRIC_AUTH_BASE_URL}/api/v1/biometric/enroll?${params.toString()}`, {
    method: "POST",
    body: forwardForm,
    headers: biometricAuthHeaders(),
  });
  return relay(backendResponse);
}
