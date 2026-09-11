# Memory / Context Log — SyncVerity

Purpose: running log of project state so a new session (human or AI agent) can
pick up context quickly without re-reading the entire conversation history.
Update this file at the end of any substantial work session.

## Current phase
Phases 0–8 scaffolded in code; full training pipeline implemented. Actual
training on FakeAVCeleb not yet run (no dataset downloaded yet).

## Host environment (as of this session)
- OS: Windows; Python 3.14.5.
- **GPU enabled:** torch 2.11.0+cu128 + torchvision/torchaudio 0.26/2.11 (+cu128)
  installed over the original CPU build — `torch.cuda.is_available()` → True.
  Driver: NVIDIA 610.88 (CUDA 13.3), GPU: RTX 5050 Laptop (Blackwell → needs
  cu128+, NOT cu121). Verified: full combined model forward/backward on GPU.
- **numpy downgraded to 2.4.6** — mandatory; numba (librosa dep) fails at import
  with numpy ≥ 2.5.
- **ffmpeg installed** via `winget install Gyan.FFmpeg` (9.0.1). Path applies to
  new terminals; the running shell needs the WINGET full path.
  `src/extraction/extract_audio.py` now locates ffmpeg via `shutil.which` with a
  winget-Packages fallback and decodes via `pcm_f32le` raw stream (replaces a
  broken torchaudio utf8→decode path; torchaudio 2.11 wants `torchcodec`, which is
  NOT installed).
- Verified on real media: 3 s synthetic mp4 → librosa+ffmpeg decode 48000 samples,
  cv2 sample_frames → 15 frames.

## Project state (as of this log)
- Full project scaffold created per `architecture.md` (dirs + `configs/config.yaml`).
- All phase modules implemented and importable:
  - Phase 1: `src/extraction/extract_video.py` (ArcFace, caches embeddings +
    aligned face crops), `extract_audio.py` (wav2vec2, caches hidden states +
    waveform).
  - Phase 2: `src/models/align.py` (time-grid pooling), `cross_attention.py`
    (bidirectional cross-attention, retains full attention weights, padding-mask
    aware). **Bug found & fixed:** `nn.MultiheadAttention` returns head-averaged
    weights by default — must pass `average_attn_weights=False` or the stacked
    attention shape collapses the wrong axis.
  - Phase 3: `src/models/classifier.py` (MLP head), `src/data/dataset.py`
    (padded grid batches + masks), `src/train.py` (modes: `fusion` / `artifact`
    / `combined`; seeding; checkpoint-every-N epochs).
  - Phase 4: `src/models/artifact_video.py` (EfficientNet on crops),
    `artifact_audio.py` (CNN on log-mel), `src/models/model.py` (composes
    branches for the ablation table).
  - Phase 5: `src/evaluate.py` (accuracy / balanced_accuracy / auroc / f1 +
    generalization drop reporting).
  - Phase 6: `src/explain/gradcam.py`, `spectrogram_highlight.py`,
    `attention_viz.py` (fixed: use `fig.canvas.buffer_rgba()`, not the removed
    `tostring_rgb()`).
  - Phase 7 (stretch): `src/reference_verification.py` (pgvector; lazy
    `psycopg2` import).
  - Phase 8 (stretch): `src/inference.py` (end-to-end analyze), FastAPI backend
    (`backend/main.py`, `backend/routers/analyze.py`, static `/results` mount),
    Next.js frontend (`frontend/` upload → verdict → explanation UI, dark
    forensic theme per `design.md`).
- `src/data/build_manifest.py` scans a dataset root and writes `manifest.csv`
  (label inferred from real/fake in path).

## Verification performed
All modules import and compile. Smoke tests passed:
- `align.py` grid pooling on synthetic 6 s video/audio sequences.
- Cross-attention + MLP head overfits a 2-sample micro-batch to ~0 loss; attn
  shapes `(B, G, G)` and full `(B, L, H, G, G)` correct.
- `SyncVerityModel` (combined) forward/backward through all branches.
- Dataset loads fabricated cache, pads to fixed grid, batches correctly.
- Explainability: attention matrix images, top spans, Grad-CAM heatmaps
  (EfficientNet 112×112 → 4×4 spatial), spectrogram overlay all produce images.
- NOTE: earlier `import cv2` failure in one check did not reproduce; all deps
  (torch, torchaudio, transformers, insightface, timm, librosa, sklearn,
  matplotlib, fastapi, opencv) import OK in this environment. On Python 3.14
  some packages may need a 3.10–3.11 venv if compilation issues appear.

## Decisions made
- Pass `average_attn_weights=False` on every cross-attention call to keep
  per-head weights for explainability.
- Dataset returns **fixed-length padded batches** (grid padded to
  `max_grid_steps`, crops to `max_crops`, waveform to `max_samples`) so the
  default collator `DataLoader` works without custom collation.
- Artifact video branch trains/caches on aligned face crops saved during ArcFace
  extraction (`_video.pt` now stores `crops`), avoiding re-decoding in the
  training loop.
- `max_grid_steps: 128`, `max_crops: 100`, `max_samples: 192000` (12 s @ 16 kHz)
  added to `config.yaml`; `paths.model_checkpoint` defaults to
  `checkpoints/syncverity_combined_best.pt`.
- Explainability images are saved under `results/<clip_id>/` and exposed as
  relative paths so FastAPI serves them at `/results/...`.

## Open decisions
- Exact frame-sampling rate for video embeddings (config default 5 fps).
- Whether to add a `--max-clips` dev subset flag to training for CPU quick
  iteration (recommended before full GPU run).
- Literature-check wording for the novelty claim (still to be drafted).

## Next steps
1. Download a FakeAVCeleb subset (~20 clips) into `data/raw/fakeavceleb`.
2. `python -m src.data.build_manifest data/raw/fakeavceleb` → writes
   `data/processed/manifest.csv`.
3. Sanity-run extraction: `python -m src.extraction.extract_video <clip>` and
   `python -m src.extraction.extract_audio <clip>`.
4. Train: `python -m src.train --mode fusion`, then `--mode artifact`, then
   `--mode combined`.
5. Train classifier with the tiny-batch overfit check before full training
   (per `rules.md`).
6. Generalization test only at the end: `python -m src.evaluate
   checkpoints/syncverity_combined_epochN.pt --split generalization`.
7. Web demo: `uvicorn backend.main:app` + `npm run dev` in `frontend/` (requires
   a trained checkpoint).

## Session log template
```
### [YYYY-MM-DD] Session summary
- What was implemented:
- What worked:
- What didn't / blockers:
- Decisions made:
- Next steps:
```