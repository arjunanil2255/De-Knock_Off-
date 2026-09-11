"use client";

import { useState } from "react";

const STAGES = [
  "extracting embeddings",
  "aligning time grids",
  "fusing cross-modal signals",
  "classifying",
];

export default function UploadForm({
  onFile,
}: {
  onFile: (file: File) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [stage, setStage] = useState<number>(-1);
  const [busy, setBusy] = useState(false);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!file || busy) return;
    setBusy(true);
    setStage(0);
    const timer = setInterval(() => {
      setStage((s) => Math.min(s + 1, STAGES.length - 1));
    }, 600);
    try {
      await onFile(file);
    } finally {
      clearInterval(timer);
      setBusy(false);
      setStage(-1);
    }
  };

  return (
    <div>
      <form className="upload" onSubmit={handleSubmit}>
        <input
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button type="submit" disabled={!file || busy}>
          {busy ? "Analyzing…" : "Analyze"}
        </button>
      </form>
      {busy && (
        <ul className="stage-list" aria-live="polite">
          {STAGES.map((label, idx) => (
            <li
              key={label}
              className={idx < stage ? "done" : idx === stage ? "active" : ""}
            >
              {label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}