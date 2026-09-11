# Design System Document — SyncVerity (Web Demo)

Applies to the Phase 8 web demo (FastAPI + Next.js). Kept lightweight since the
demo's job is to clearly present a verdict and its evidence, not to be a full
product UI.

## Design intent
The demo should read as a **forensic/analysis tool**, not a consumer app — calm,
high-contrast, evidence-forward. The verdict and its explanation (heatmap,
spectrogram, attention visualization) are the content; the UI should stay out of
the way of that evidence.

## Color palette
- **Background:** near-black / dark neutral (`#0E0F12`) — keeps focus on
  video/heatmap imagery, which reads better on dark backgrounds.
- **Surface panels:** dark gray (`#1A1B1F`)
- **Primary accent (real/authentic verdict):** teal/green (`#3DDC97`)
- **Alert accent (fake/manipulated verdict):** warm red (`#E5484D`)
- **Neutral text:** off-white (`#F2F2F2`) for primary, muted gray (`#9A9AA2`) for
  secondary text
- **Attention/heatmap overlays:** use a perceptually uniform colormap (e.g.
  viridis/magma) rather than default jet, for accurate visual interpretation

## Typography
- **Headings:** a clean geometric sans (e.g. Inter or Söhne) — confidence and
  verdict labels should be large and unambiguous.
- **Body/UI text:** same family, regular weight, for consistency.
- **Monospace** (e.g. JetBrains Mono) for confidence scores, timestamps, and any
  raw metric values — reinforces the "analysis tool" feel.

## Layout principles
- Upload/results as a single-column focused flow — no dashboard clutter.
- Verdict + confidence shown immediately and prominently at the top of results.
- Explanation artifacts (Grad-CAM, spectrogram, attention viz) laid out as
  clearly labeled panels below the verdict — each with a one-line caption
  explaining what it shows.
- Reference-verification mode (if enabled) gets a visually distinct secondary
  panel, not merged into the primary verdict — it's an additional signal, not
  the main claim.

## Interaction notes
- Processing state (during inference) should show real pipeline stages
  (extracting embeddings → aligning → fusing → classifying) rather than a
  generic spinner — reinforces the tool's credibility.
- No dark-pattern styling, no manufactured urgency — this is a trust-sensitive
  tool and should read as sober and transparent.
