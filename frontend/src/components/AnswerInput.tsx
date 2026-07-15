"use client";

import { useState } from "react";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";

interface AnswerInputProps {
  onSubmitText: (text: string) => void;
  onSubmitAudio: (audio: Blob) => void;
  disabled: boolean;
}

export function AnswerInput({ onSubmitText, onSubmitAudio, disabled }: AnswerInputProps) {
  const [mode, setMode] = useState<"text" | "voice">("text");
  const [text, setText] = useState("");
  const recorder = useAudioRecorder();

  const handleTextSubmit = () => {
    if (!text.trim()) return;
    onSubmitText(text.trim());
    setText("");
  };

  const handleUseRecording = () => {
    if (recorder.audioBlob) {
      onSubmitAudio(recorder.audioBlob);
      recorder.reset();
    }
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex gap-2 text-sm">
        <button
          type="button"
          onClick={() => setMode("text")}
          className={`rounded-full px-3 py-1 ${mode === "text" ? "bg-indigo-600 text-white" : "border border-neutral-300 dark:border-neutral-600"}`}
        >
          Type
        </button>
        <button
          type="button"
          onClick={() => setMode("voice")}
          className={`rounded-full px-3 py-1 ${mode === "voice" ? "bg-indigo-600 text-white" : "border border-neutral-300 dark:border-neutral-600"}`}
        >
          Speak
        </button>
      </div>

      {mode === "text" ? (
        <div className="flex flex-col gap-2">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={5}
            disabled={disabled}
            placeholder="Type your answer…"
            className="rounded-md border border-neutral-300 px-3 py-2 text-sm dark:border-neutral-600 dark:bg-neutral-900"
          />
          <button
            type="button"
            onClick={handleTextSubmit}
            disabled={disabled || !text.trim()}
            className="self-start rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
          >
            Submit answer
          </button>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {recorder.status === "unsupported" && (
            <p className="text-sm text-red-600">Voice recording isn&apos;t supported in this browser.</p>
          )}
          {recorder.status === "denied" && (
            <p className="text-sm text-red-600">Microphone access was denied.</p>
          )}

          <div className="flex items-center gap-3">
            {recorder.status !== "recording" ? (
              <button
                type="button"
                onClick={recorder.start}
                disabled={disabled}
                className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
              >
                {recorder.audioBlob ? "Record again" : "Start recording"}
              </button>
            ) : (
              <button
                type="button"
                onClick={recorder.stop}
                className="rounded-md bg-red-600 px-4 py-2 text-sm font-medium text-white"
              >
                Stop recording
              </button>
            )}
            {recorder.status === "recording" && <span className="text-sm text-red-600">● Recording…</span>}
          </div>

          {recorder.audioBlob && recorder.status === "stopped" && (
            <div className="flex items-center gap-3">
              <audio controls src={URL.createObjectURL(recorder.audioBlob)} className="h-9" />
              <button
                type="button"
                onClick={handleUseRecording}
                disabled={disabled}
                className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
              >
                Submit this recording
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
