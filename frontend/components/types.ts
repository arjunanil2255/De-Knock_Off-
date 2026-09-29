export type AnalysisResult = {
  clip_id: string;
  verdict: "real" | "fake";
  confidence: number;
  scores: {
    fusion_embedding_norm: number;
    video_artifact: number;
    audio_artifact: number;
  };
  explanation_paths?: Record<string, string>;
  attention_spans?: {
    video_to_audio: Span[];
    audio_to_video: Span[];
  };
};

export type Span = { query_time: number; key_time: number; weight: number };

export const API_BASE = "http://127.0.0.1:8000";
export const ASSET_BASE = `${API_BASE}/results`;
export const darkStyleKey = "forensic-dark";
