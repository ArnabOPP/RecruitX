"use client";

import { useEffect, useState } from "react";
import { useWebcam } from "@/hooks/useWebcam";

interface WebcamCaptureProps {
  /** How many still frames the caller needs (e.g. 3 for enrollment). */
  targetCount: number;
  onComplete: (frames: Blob[]) => void;
  label: string;
}

/** A handful of real still frames captured on demand — not a
 * client-computed liveness/identity verdict, just raw JPEG bytes for the
 * server to actually verify. Matches the "a few real frames per check"
 * design this whole platform is built around. */
export function WebcamCapture({ targetCount, onComplete, label }: WebcamCaptureProps) {
  const { videoRef, status, errorMessage, start, stop, captureFrame } = useWebcam();
  const [frames, setFrames] = useState<Blob[]>([]);
  const [thumbnails, setThumbnails] = useState<string[]>([]);

  useEffect(() => {
    start();
    return () => stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    return () => thumbnails.forEach((url) => URL.revokeObjectURL(url));
  }, [thumbnails]);

  const handleCapture = async () => {
    const frame = await captureFrame();
    if (!frame) return;
    const next = [...frames, frame];
    setFrames(next);
    setThumbnails((prev) => [...prev, URL.createObjectURL(frame)]);
    if (next.length >= targetCount) onComplete(next);
  };

  const handleReset = () => {
    setFrames([]);
    setThumbnails((prev) => {
      prev.forEach((url) => URL.revokeObjectURL(url));
      return [];
    });
  };

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative aspect-video w-full max-w-md overflow-hidden rounded-lg bg-black/80">
        <video ref={videoRef} muted playsInline className="h-full w-full -scale-x-100 object-cover" />
        {status !== "active" && (
          <div className="absolute inset-0 flex items-center justify-center text-center text-sm text-white/80">
            {status === "requesting" && "Requesting camera access…"}
            {status === "denied" && "Camera access was denied. Allow it in your browser settings and reload."}
            {status === "unsupported" && "This browser doesn't support camera capture."}
            {status === "error" && (errorMessage ?? "Couldn't access the camera.")}
            {status === "idle" && "Starting camera…"}
          </div>
        )}
      </div>

      <p className="text-sm text-neutral-500">
        {label} — {frames.length}/{targetCount} captured
      </p>

      {thumbnails.length > 0 && (
        <div className="flex gap-2">
          {thumbnails.map((url) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img key={url} src={url} alt="captured frame" className="h-14 w-14 rounded object-cover" />
          ))}
        </div>
      )}

      <div className="flex gap-3">
        <button
          type="button"
          onClick={handleCapture}
          disabled={status !== "active" || frames.length >= targetCount}
          className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
        >
          Capture frame
        </button>
        {frames.length > 0 && (
          <button
            type="button"
            onClick={handleReset}
            className="rounded-md border border-neutral-300 px-4 py-2 text-sm font-medium text-neutral-700 dark:border-neutral-600 dark:text-neutral-200"
          >
            Retake
          </button>
        )}
      </div>
    </div>
  );
}
