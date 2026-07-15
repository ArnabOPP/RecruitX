import { NextRequest } from "next/server";
import { ORCHESTRATOR_BASE_URL, orchestratorHeaders, relay } from "@/lib/backend";

export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const body = await request.json();

  const backendResponse = await fetch(`${ORCHESTRATOR_BASE_URL}/api/v1/sessions/${id}/answer`, {
    method: "POST",
    headers: { "content-type": "application/json", ...orchestratorHeaders() },
    body: JSON.stringify(body),
  });
  return relay(backendResponse);
}
