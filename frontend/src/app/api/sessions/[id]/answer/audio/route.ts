import { NextRequest } from "next/server";
import { ORCHESTRATOR_BASE_URL, orchestratorHeaders, relay } from "@/lib/backend";

export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const form = await request.formData();

  const backendResponse = await fetch(`${ORCHESTRATOR_BASE_URL}/api/v1/sessions/${id}/answer/audio`, {
    method: "POST",
    body: form,
    headers: orchestratorHeaders(),
  });
  return relay(backendResponse);
}
