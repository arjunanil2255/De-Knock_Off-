# Product Requirement Document — SyncVerity

## Project
**Working title:** SyncVerity
**Full title:** Cross-Modal Consistency-Based Deepfake Detection Using Vector Similarity Scoring
**Type:** B.Tech capstone project (AI specialization)

## Vision
Most deepfake detectors judge video and audio in isolation, overfit to the specific
generation method they were trained on, and give no explanation for their verdict.
SyncVerity instead asks a different question: **does the face and the voice in a clip
genuinely correspond to each other over time?** A real clip has one underlying source
of truth; a deepfake is usually a stitch of mismatched pieces. By learning this
cross-modal correspondence with attention (rather than a fixed similarity formula),
and pairing every verdict with a visual explanation, SyncVerity targets the three gaps
that limit current tools: single-modality analysis, poor generalization, and lack of
interpretability.

## Problem statement
- Deepfakes are cheap and fast to produce; humans detect them only ~0.1% of the time.
- Most automated detectors are single-modality (video-only or audio-only).
- Most detectors overfit to the manipulation technique seen in training and fail on
  unseen techniques.
- Most detectors output a bare score with no explanation, which limits trust and
  adoption in real investigative or moderation workflows.

## Target users
- **Primary (capstone context):** academic evaluators/reviewers assessing technical
  novelty and rigor.
- **Illustrative real-world users** (for framing, not near-term deployment):
  - Content moderation teams at platforms needing a first-pass deepfake flag.
  - Journalists/fact-checkers verifying suspect clips of public figures.
  - Organizations wanting to verify a clip against a known-authentic reference
    (e.g. confirming a CEO video message is genuine).

## Core features (MVP scope for capstone)
1. **Multimodal ingestion** — accepts a video clip, splits into frame stream + audio track.
2. **Embedding extraction** — ArcFace for face/lip embeddings, wav2vec2 for voice embeddings.
3. **Per-modality artifact baseline** — CNN (Xception/EfficientNet) for visual artifacts,
   spoof classifier for audio artifacts.
4. **Cross-attention fusion (core novelty)** — transformer cross-attention between video
   and audio embedding sequences, producing a consistency signal.
5. **Fusion classifier** — MLP combining consistency signal + artifact scores → real/fake
   verdict with confidence.
6. **Explainability layer** — Grad-CAM heatmap (video), spectrogram highlight (audio),
   attention-weight visualization (cross-modal misalignment).
7. **Reference-verification mode (stretch)** — compare suspect clip against a known-authentic
   reference clip via vector similarity (pgvector) to catch identity impersonation.
8. **Web demo** — FastAPI backend + Next.js frontend for upload and results display.

## Out of scope (for this capstone cycle)
- Real-time/streaming detection.
- Mobile app.
- Production-grade scaling, auth, multi-tenant deployment.
- Support for manipulation types outside face/voice swap and reenactment
  (e.g. full-body deepfakes) unless time permits.

## Success criteria
- Cross-attention fusion model trains end-to-end on FakeAVCeleb and outperforms the
  per-modality-only baseline on held-out FakeAVCeleb data.
- Model shows measurably better generalization (smaller accuracy drop) on
  AV-Deepfake1M / LAV-DF than a naive single-modality baseline.
- Explainability outputs are produced for every verdict, not just aggregate metrics.
- Working end-to-end demo: upload clip → verdict → explanation, via the web UI.

## Datasets
- **Train/val:** FakeAVCeleb
- **Generalization test (held out, unseen manipulation methods):** AV-Deepfake1M / LAV-DF
