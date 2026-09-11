# Rules & Standards Document — SyncVerity

## Purpose
Guidelines for anyone (human or AI coding agent) working on this codebase, to keep
implementation consistent with `architecture.md` and avoid common failure modes in
multimodal deepfake detection projects.

## General coding standards
- Python: PEP8, type hints on all function signatures, docstrings on public functions.
- One responsibility per module — extraction, alignment, fusion, classification, and
  explainability stay in separate files (see `architecture.md` folder layout). Do not
  collapse these into a single monolithic script.
- All randomness (data splits, model init) must be seeded for reproducibility.
- No hardcoded file paths — read from `configs/config.yaml`.
- Config-driven hyperparameters (learning rate, batch size, `d_model`, number of
  attention heads, etc.) — never hardcode inside training loops.

## Data handling
- Never commit raw dataset files (`data/raw/`) to version control — add to `.gitignore`.
- Cache all extracted embeddings to `data/processed/` as `.pt` files, keyed by clip ID.
  Re-extracting embeddings every run is not acceptable; check cache first.
- Keep FakeAVCeleb (train/val) and AV-Deepfake1M/LAV-DF (generalization test) in
  physically separate directories — never let generalization-test data leak into
  training or validation splits, even accidentally.
- Log dataset split sizes and class balance at the start of every training run.

## Modeling practices
- Preserve attention weights from `cross_attention.py` on every forward pass — the
  explainability layer depends on these; do not discard them for "efficiency" during
  early development.
- The per-modality artifact baseline and the cross-attention fusion path must be
  evaluated **both separately and combined** — the project's core claim rests on
  showing the fused approach outperforms either baseline alone. Never skip the
  baseline-only ablation.
- Any new model component must sanity-check first: overfit a tiny batch (as small as
  8–16 samples) and confirm loss drives toward zero before scaling to the full dataset.
- Do not change the evaluation dataset (AV-Deepfake1M/LAV-DF) or its split until the
  final generalization test — using it for iteration is a data leakage risk that
  invalidates the generalization claim.

## Error handling
- All extraction scripts must handle missing faces / silent audio gracefully (skip
  and log, don't crash the batch).
- All API endpoints (FastAPI) must return structured error responses, not raw
  stack traces, to the frontend.
- Training scripts must checkpoint regularly (every N epochs) — do not rely on a
  single run completing without interruption.

## Anti-patterns to avoid
- Do not recompute embeddings inside the training loop.
- Do not mix video/audio sampling rates without going through the alignment step.
- Do not report accuracy on FakeAVCeleb alone as the headline result — always pair
  it with the generalization-test result, since that's the project's central claim.
- Do not add the reference-verification (pgvector) mode or the web demo before the
  core fusion model is validated — sequencing matters (see `phase.md`).

## AI agent boundaries
- An AI coding agent working on this repo should not silently change the model
  architecture (e.g. attention head count, embedding dimensions) without flagging
  the change — these are tracked decisions.
- An AI coding agent should not delete or overwrite cached embeddings without
  explicit instruction.
- An AI coding agent should update `memory.md` with what was implemented at the end
  of any substantial session before ending that session.
