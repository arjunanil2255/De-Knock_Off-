"use client";

import { AnalysisResult, ASSET_BASE, Span } from "./types";

function SpansTable({ spans, label }: { spans: Span[]; label: string }) {
  if (!spans || spans.length === 0) return null;
  return (
    <div className="spans">
      <h3>{label}</h3>
      <table>
        <thead>
          <tr>
            <th>query (s)</th>
            <th>key (s)</th>
            <th>weight</th>
          </tr>
        </thead>
        <tbody>
          {spans.map((span, idx) => (
            <tr key={idx}>
              <td>{span.query_time.toFixed(2)}</td>
              <td>{span.key_time.toFixed(2)}</td>
              <td>{span.weight.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ResultView({ result }: { result: AnalysisResult }) {
  const paths = result.explanation_paths ?? {};

  const explanationPanels = [
    { key: "attention_video_to_audio", title: "Video → Audio attention", caption: "For each video instant, the audio instants the fusion model weighted most strongly." },
    { key: "attention_audio_to_video", title: "Audio → Video attention", caption: "For each audio instant, the video instants the fusion model weighted most strongly. Attention is supporting evidence, not a calibrated mismatch score." },
    { key: "audio_spectrogram_highlight", title: "Audio spectrogram", caption: "Log-mel spectrogram with model-focused spectral regions highlighted." },
    ...Object.entries(paths)
      .filter(([key]) => key.startsWith("gradcam_frame_"))
      .map(([key], idx) => ({
        key,
        title: `Face frame ${idx}`,
        caption: "Grad-CAM heatmap of the artifact branch for the final predicted class over the aligned face crop.",
      })),
  ].filter((p) => paths[p.key]);

  return (
    <div>
      <div className="card">
        <div className={`verdict ${result.verdict}`}>
          {result.verdict === "real" ? "AUTHENTIC" : "MANIPULATED"}
          <span className="confidence">
            {`${(result.confidence * 100).toFixed(1)}% confidence`}
          </span>
        </div>
        <div className="scores">
          <div className="score">
            <span className="label">fusion embedding norm (diagnostic)</span>
            <span className="value">{result.scores.fusion_embedding_norm.toFixed(3)}</span>
          </div>
          <div className="score">
            <span className="label">video artifact</span>
            <span className="value">{result.scores.video_artifact.toFixed(3)}</span>
          </div>
          <div className="score">
            <span className="label">audio artifact</span>
            <span className="value">{result.scores.audio_artifact.toFixed(3)}</span>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>Explanation</h2>
        <div className="explanations">
          {explanationPanels.map((panel) => (
            <figure className="explanation" key={panel.key}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`${ASSET_BASE}/${paths[panel.key]}`}
                alt={panel.title}
              />
              <figcaption className="caption">
                <strong>{panel.title}:</strong> {panel.caption}
              </figcaption>
            </figure>
          ))}
        </div>
      </div>

      {result.attention_spans && (
        <div className="card">
          <h2>Top cross-modal alignments</h2>
          <div className="scores">
            <SpansTable spans={result.attention_spans.video_to_audio} label="Video → Audio" />
            <SpansTable spans={result.attention_spans.audio_to_video} label="Audio → Video" />
          </div>
        </div>
      )}
    </div>
  );
}
