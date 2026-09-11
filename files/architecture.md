# Architecture Document — SyncVerity

## App flow (inference path)
1. User uploads a video clip via the Next.js frontend.
2. FastAPI backend receives the file, splits it into frame stream + audio track.
3. **Video branch:** face detection/alignment → ArcFace embeddings (T_v × 512),
   sampled at ~5–10 fps.
4. **Audio branch:** wav2vec2 embeddings (T_a × 768), pooled to the same time grid
   as the video branch.
5. **Baseline artifact branch (parallel):** Xception/EfficientNet CNN scores video
   frames for blending artifacts; spoof classifier scores audio for spectral artifacts.
6. **Alignment step:** interpolate/pool both embedding sequences onto a common time
   grid (e.g. 10 steps/sec) before fusion.
7. **Cross-attention fusion module:** video queries audio, audio queries video;
   produces attended representations + attention weight matrices (kept for
   explainability).
8. **Fusion classifier (MLP):** consistency signal + two artifact scores → real/fake
   logit + confidence.
9. **Explainability layer:** Grad-CAM (video), spectrogram highlight (audio),
   attention-weight visualization (temporal misalignment) generated from the same
   forward pass.
10. **(Optional) Reference-verification:** if a reference clip is supplied, embeddings
    are compared via pgvector similarity search against the suspect clip.
11. Backend returns verdict + confidence + explanation assets to the frontend for display.

## Structural architecture

```
syncverity/
├── data/
│   ├── raw/                     # FakeAVCeleb / AV-Deepfake1M clips (not committed)
│   └── processed/                # cached embeddings (.pt)
├── src/
│   ├── extraction/
│   │   ├── extract_video.py      # face detect/align -> ArcFace embeddings
│   │   └── extract_audio.py      # wav2vec2 embeddings
│   ├── models/
│   │   ├── align.py              # time-alignment / pooling utilities
│   │   ├── cross_attention.py    # fusion module (core novelty)
│   │   ├── artifact_video.py     # Xception/EfficientNet baseline
│   │   ├── artifact_audio.py     # audio spoof classifier baseline
│   │   └── classifier.py         # MLP fusion head
│   ├── explain/
│   │   ├── gradcam.py
│   │   ├── spectrogram_highlight.py
│   │   └── attention_viz.py
│   ├── data/
│   │   └── dataset.py            # PyTorch Dataset/Dataloader
│   ├── train.py
│   ├── evaluate.py               # generalization test on AV-Deepfake1M/LAV-DF
│   └── reference_verification.py # pgvector similarity search
├── backend/                      # FastAPI app
│   ├── main.py
│   └── routers/
├── frontend/                     # Next.js app
│   ├── app/
│   └── components/
├── configs/
│   └── config.yaml
├── prd.md
├── architecture.md
├── rules.md
├── phase.md
├── design.md
└── memory.md
```

## Tech stack
| Layer | Choice | Notes |
|---|---|---|
| Video embeddings | ArcFace (insightface) | face/lip identity + motion fingerprint |
| Audio embeddings | wav2vec2 (HuggingFace) | speaker identity + vocal characteristics |
| Video artifact baseline | Xception or EfficientNet | pretrained, fine-tuned |
| Audio artifact baseline | Spoof classifier (spectral) | pretrained or trained from scratch |
| Fusion | Custom cross-attention (PyTorch `nn.MultiheadAttention`) | core novel component |
| Classifier head | MLP | small, fast to iterate on |
| Vector similarity | pgvector (Postgres extension) | reference-verification mode |
| Backend | FastAPI | serves inference pipeline |
| Frontend | Next.js | upload + results UI |
| Training/eval datasets | FakeAVCeleb (train/val), AV-Deepfake1M / LAV-DF (generalization test) | |

## Key technical decisions to lock early
- **Sampling rate mismatch:** video (~25–30 fps) vs. wav2vec2 output rate differ —
  resolved via pooling/interpolation to a shared time grid before fusion (see
  `src/models/align.py`). This is the most error-prone step; test it in isolation
  with unit tests on known-length inputs before wiring into the full pipeline.
- **Embedding caching:** ArcFace/wav2vec2 extraction is expensive — always cache to
  `data/processed/*.pt`, never recompute per epoch.
- **Attention weights are retained from the first implementation**, even before the
  explainability milestone is built, to avoid retrofitting.
