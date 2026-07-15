import { ORCHESTRATOR_BASE_URL, orchestratorHeaders, relay } from "@/lib/backend";

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  const backendResponse = await fetch(`${ORCHESTRATOR_BASE_URL}/api/v1/sessions/${id}/report`, {
    headers: orchestratorHeaders(),
    cache: "no-store",
  });
  return relay(backendResponse);
}
