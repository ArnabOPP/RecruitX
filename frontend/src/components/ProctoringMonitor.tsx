"use client";

import { useEffect, useRef, useState } from "react";
import { useWebcam } from "@/hooks/useWebcam";
import { submitProctoringSnapshot } from "@/lib/api-client";

interface ProctoringMonitorProps {
  sessionId: string;
  active: boolean;
  intervalMs: number;
}

/** A small always-on proctoring widget: captures a real frame on an
 * interval and submits it to the real proctoring service, showing only
 * what the server actually decided (integrity score, flags) — never a
 * client-computed verdict. Runs for the lifetime of the interview page,
 * independent of which question/round is currently showing. */
export function ProctoringMonitor({ sessionId, active, intervalMs }: ProctoringMonitorProps) {
  const { videoRef, status, start, stop, captureFrame } = useWebcam();
  const [integrityScore, setIntegrityScore] = useState<number | null>(null);
  const [lastFlags, setLastFlags] = useState<string[]>([]);
  const [framesSent, setFramesSent] = useState(0);
  const inFlightRef = useRef(false);

  useEffect(() => {
    if (active) start();
    else stop();
    return () => stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  useEffect(() => {
    if (!active || status !== "active") return;

    const timer = setInterval(async () => {
      if (inFlightRef.current) return;
      const frame = await captureFrame();
      if (!frame) return;
      inFlightRef.current = true;
      try {
        const result = await submitProctoringSnapshot(sessionId, frame);
        setIntegrityScore(result.integrity_score);
        setLastFlags(result.flagged_this_frame);
        setFramesSent(result.frames_processed);
      } catch {
        // A missed snapshot isn't fatal — the next interval tick tries again.
      } finally {
        inFlightRef.current = false;
      }
    }, intervalMs);

    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, status, intervalMs, sessionId]);

  if (!active) return null;

  return (
    <div className="flex items-center gap-3 rounded-md border border-neutral-200 bg-neutral-50 px-3 py-2 text-xs dark:border-neutral-700 dark:bg-neutral-900">
      <div className="relative h-14 w-20 overflow-hidden rounded bg-black">
        <video ref={videoRef} muted playsInline className="h-full w-full -scale-x-100 object-cover" />
      </div>
      <div className="flex flex-col gap-0.5">
        <span className="font-medium">
          Proctoring {status === "active" ? "active" : status === "denied" ? "camera denied" : "starting…"}
        </span>
        {integrityScore !== null && (
          <span className="text-neutral-500">
            Integrity score: {integrityScore.toFixed(0)} · {framesSent} frame(s) checked
          </span>
        )}
        {lastFlags.length > 0 && <span className="text-amber-600">Flagged: {lastFlags.join(", ")}</span>}
      </div>
    </div>
  );
}
