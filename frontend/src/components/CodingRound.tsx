"use client";

import { useState } from "react";
import { submitCode } from "@/lib/api-client";
import { ApiError, CodeSubmissionResponse } from "@/lib/types";

interface TestCase {
  input: string;
  expectedOutput: string;
}

interface CodingRoundProps {
  sessionId: string;
  onCompleted: (result: CodeSubmissionResponse) => void;
}

const LANGUAGES = ["python", "javascript"];

export function CodingRound({ sessionId, onCompleted }: CodingRoundProps) {
  const [language, setLanguage] = useState("python");
  const [sourceCode, setSourceCode] = useState("");
  const [testCases, setTestCases] = useState<TestCase[]>([{ input: "", expectedOutput: "" }]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateTestCase = (index: number, field: keyof TestCase, value: string) => {
    setTestCases((prev) => prev.map((tc, i) => (i === index ? { ...tc, [field]: value } : tc)));
  };

  const addTestCase = () => setTestCases((prev) => [...prev, { input: "", expectedOutput: "" }]);
  const removeTestCase = (index: number) => setTestCases((prev) => prev.filter((_, i) => i !== index));

  const handleSubmit = async () => {
    setError(null);
    const cases = testCases.filter((tc) => tc.expectedOutput.trim().length > 0);
    if (!sourceCode.trim()) {
      setError("Write a solution before submitting.");
      return;
    }
    if (cases.length === 0) {
      setError("Add at least one test case with an expected output.");
      return;
    }
    setSubmitting(true);
    try {
      const result = await submitCode(sessionId, {
        language,
        sourceCode,
        testCases: cases.map((tc) => ({ input: tc.input || undefined, expected_output: tc.expectedOutput })),
      });
      onCompleted(result);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Code submission failed.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <label className="flex flex-col gap-1.5 text-sm">
        <span className="font-medium">Language</span>
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="w-40 rounded-md border border-neutral-300 px-3 py-2 dark:border-neutral-600 dark:bg-neutral-900"
        >
          {LANGUAGES.map((lang) => (
            <option key={lang} value={lang}>
              {lang}
            </option>
          ))}
        </select>
      </label>

      <label className="flex flex-col gap-1.5 text-sm">
        <span className="font-medium">Solution</span>
        <textarea
          value={sourceCode}
          onChange={(e) => setSourceCode(e.target.value)}
          rows={14}
          placeholder="Write your solution here…"
          spellCheck={false}
          className="rounded-md border border-neutral-300 px-3 py-2 font-mono text-sm dark:border-neutral-600 dark:bg-neutral-900"
        />
      </label>

      <div className="flex flex-col gap-2">
        <span className="text-sm font-medium">Test cases</span>
        {testCases.map((tc, i) => (
          <div key={i} className="flex items-start gap-2">
            <input
              value={tc.input}
              onChange={(e) => updateTestCase(i, "input", e.target.value)}
              placeholder="stdin input (optional)"
              className="flex-1 rounded-md border border-neutral-300 px-2 py-1.5 text-sm dark:border-neutral-600 dark:bg-neutral-900"
            />
            <input
              value={tc.expectedOutput}
              onChange={(e) => updateTestCase(i, "expectedOutput", e.target.value)}
              placeholder="expected output"
              className="flex-1 rounded-md border border-neutral-300 px-2 py-1.5 text-sm dark:border-neutral-600 dark:bg-neutral-900"
            />
            {testCases.length > 1 && (
              <button
                type="button"
                onClick={() => removeTestCase(i)}
                className="px-2 text-sm text-neutral-400 hover:text-red-600"
              >
                ✕
              </button>
            )}
          </div>
        ))}
        <button type="button" onClick={addTestCase} className="self-start text-sm text-indigo-600">
          + Add test case
        </button>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <button
        type="button"
        onClick={handleSubmit}
        disabled={submitting}
        className="self-start rounded-md bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white disabled:opacity-40"
      >
        {submitting ? "Grading…" : "Submit solution"}
      </button>
    </div>
  );
}
