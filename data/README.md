# Dataset setup

Do not commit raw media to this repository. The required datasets have
research-access conditions and must be obtained by the project owner.

## 1. FakeAVCeleb: train and validation

Request access from the [official FakeAVCeleb download page](https://sites.google.com/view/fakeavcelebdash-lab/download).
After approval, unpack the supplied download into:

```text
data/raw/fakeavceleb/
```

Create the train/validation manifest and cached features:

```powershell
python -m src.data.build_manifest data/raw/fakeavceleb
python -m src.data.extract_manifest --device cuda
```

Confirm that the generated `data/processed/manifest.csv` has both `train` and
`val` rows before training. If source identities are encoded differently from
the default filename prefix, set `--source-group-pattern` to a regular
expression with capture group 1 containing the source identity.

Before a full run, use a small reproducible subset to validate the wiring:

```powershell
python -m src.train --mode fusion --max-clips 16 --epochs 20
```

Only proceed when this small run can overfit as expected; then train the
fusion, artifact, and combined modes without `--max-clips`.

## 2. AV-Deepfake1M: held-out generalization evaluation

Obtain access under the dataset's official EULA from the
[AV-Deepfake1M repository](https://github.com/ControlNet/AV-Deepfake1M).
Unpack it separately:

```text
data/raw/av_deepfake1m/
```

Never mix it into the FakeAVCeleb manifest. Build and extract a separate
generalization manifest:

```powershell
python -m src.data.build_avdeepfake_manifest data/raw/av_deepfake1m `
  --output data/processed/av_deepfake1m_manifest.csv
python -m src.data.extract_manifest `
  --manifest data/processed/av_deepfake1m_manifest.csv `
  --splits generalization `
  --device cuda
```

After training, compare the final model without tuning on this dataset:

```powershell
python -m src.evaluate checkpoints/syncverity_combined_best.pt `
  --split generalization `
  --manifest data/processed/av_deepfake1m_manifest.csv `
  --compare-split val `
  --compare-manifest data/processed/manifest.csv
```

The evaluation command writes a JSON report under `results/`.
