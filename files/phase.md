# Phases / Roadmap Document — SyncVerity

## Phase 0 — Setup
- Environment setup (PyTorch, transformers, insightface, librosa, etc.)
- Project scaffold created per `architecture.md`
- FakeAVCeleb subset downloaded and sanity-checked (~20 clips)
- Quick literature check to position the novelty claim clearly (what's genuinely new
  vs. published cross-attention deepfake-detection work)

**Exit criteria:** environment runs, sample clips load, extraction scripts execute
without error on a handful of clips.

## Phase 1 — Embedding extraction pipeline
- `extract_video.py`: face detect/align → ArcFace embeddings
- `extract_audio.py`: wav2vec2 embeddings
- Cache both to `data/processed/`
- Sanity-check embedding shapes and value ranges

**Exit criteria:** embeddings cached for full FakeAVCeleb sample set; shapes verified.

## Phase 2 — Alignment + cross-attention fusion (core novelty)
- `align.py`: time-alignment/pooling between video and audio sequences
- `cross_attention.py`: fusion module, retaining attention weights
- Overfit test on a tiny batch (8–16 samples) to confirm the wiring works

**Exit criteria:** loss drives to near-zero on the tiny overfit batch; attention
weights are inspectable.

## Phase 3 — Fusion classifier + baseline training on FakeAVCeleb
- `classifier.py`: MLP head on fused representation
- Full train/val split on FakeAVCeleb
- Train and evaluate the fusion model

**Exit criteria:** fusion model trained end-to-end; validation accuracy/metrics recorded.

## Phase 4 — Per-modality artifact baseline + ablation
- `artifact_video.py`, `artifact_audio.py` implemented
- Combine artifact scores with fusion signal in the classifier
- Run the required ablation: baseline-only vs. fusion-only vs. combined

**Exit criteria:** ablation table showing combined approach outperforms either
baseline alone (or an honest account of where it doesn't).

## Phase 5 — Generalization evaluation
- `evaluate.py` run on AV-Deepfake1M / LAV-DF (held out, untouched until now)
- Compare generalization drop for fusion approach vs. single-modality baseline

**Exit criteria:** generalization results recorded and compared to Phase 3/4 results.

## Phase 6 — Explainability layer
- `gradcam.py`, `spectrogram_highlight.py`, `attention_viz.py`
- Wire explainability outputs to real verdicts from the trained model

**Exit criteria:** for any given clip, the system produces verdict + all three
explanation artifacts.

## Phase 7 (stretch) — Reference-verification mode
- pgvector setup, `reference_verification.py`
- Compare suspect clip against known-authentic reference via similarity search

**Exit criteria:** given a reference clip + suspect clip, system returns an
impersonation-likelihood signal.

## Phase 8 (stretch) — Web demo
- FastAPI backend wiring the full inference pipeline
- Next.js frontend for upload + results + explanation display

**Exit criteria:** end-to-end working demo, upload → verdict → explanation, in browser.

## Notes on sequencing
- Phases 0–4 are the non-negotiable core of the capstone's novelty claim.
- Phase 5 (generalization) should not be touched for iteration/tuning — it's the
  final held-out test, per `rules.md`.
- Phases 7–8 are valuable for demo quality but should be the first things cut if
  time runs short.
