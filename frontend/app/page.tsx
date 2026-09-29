"use client";

import { useState } from "react";
import ResultView from "@/components/ResultView";
import UploadForm from "@/components/UploadForm";
import { AnalysisResult, API_BASE } from "@/components/types";

export default function Home() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const analyzeFile = async (file: File) => {
    setError(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        const payload = (await response.json()) as {
          detail?: string | { message?: string };
        };
        const message =
          typeof payload.detail === "string"
            ? payload.detail
            : payload.detail?.message;
        throw new Error(message ?? `Backend error (${response.status})`);
      }
      setResult((await response.json()) as AnalysisResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    }
  };

  return (
    <main>
      <header>
        <h1>SyncVerity</h1>
        <p className="subtitle">
          Cross-modal consistency deepfake detection · face vs. voice alignment
        </p>
      </header>

      <div className="card">
        <UploadForm onFile={analyzeFile} />
      </div>

      {error && <p className="error">{error}</p>}
      {result && <ResultView result={result} />}
    </main>
  );
}
