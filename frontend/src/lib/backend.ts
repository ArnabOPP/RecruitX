/** Server-only backend configuration and a shared proxy helper for the
 * Next.js Route Handlers under src/app/api/. Every browser-facing call
 * goes through one of those routes rather than hitting the orchestrator
 * or biometric-auth directly — the Next.js server talks to them
 * server-to-server, so the browser never needs CORS configured on either
 * backend service at all. */

import "server-only";
import { ApiErrorBody } from "./types";

export const ORCHESTRATOR_BASE_URL = process.env.ORCHESTRATOR_BASE_URL ?? "http://localhost:8004";
export const BIOMETRIC_AUTH_BASE_URL = process.env.BIOMETRIC_AUTH_BASE_URL ?? "http://localhost:8005";

function authHeaders(apiKey: string | undefined): HeadersInit {
  return apiKey ? { "X-API-Key": apiKey } : {};
}

export const orchestratorHeaders = () => authHeaders(process.env.ORCHESTRATOR_API_KEY);
export const biometricAuthHeaders = () => authHeaders(process.env.BIOMETRIC_AUTH_API_KEY);

/** Forwards a backend Response straight through to the browser, preserving
 * status code and JSON body — including error bodies, since the backend's
 * {error, detail, request_id} shape is what the UI already knows how to
 * render. */
export async function relay(backendResponse: Response): Promise<Response> {
  const text = await backendResponse.text();
  return new Response(text, {
    status: backendResponse.status,
    headers: { "content-type": backendResponse.headers.get("content-type") ?? "application/json" },
  });
}

export function isErrorBody(value: unknown): value is ApiErrorBody {
  return typeof value === "object" && value !== null && "error" in value && "detail" in value;
}
