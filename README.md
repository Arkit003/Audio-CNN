# Audio-CNN

Environmental sound classification with a residual CNN trained on **ESC-50**, hosted on **Modal**, and inspected through a small Next.js visualizer. This repo is about the **audio CNN workflow** — data → spectrogram → model → class probabilities + intermediate activations — not frontend engineering.

```
wav upload ──▶ mono + resample ──▶ mel-spectrogram ──▶ AudioCNN ──▶ top-3 predictions
                                                        │
                                                        └──▶ feature maps + spectrogram ──▶ visualizer
```

## What This Repo Contains

| Path | Purpose |
| --- | --- |
| `model.py` | Residual CNN architecture (`AudioCNN`) with optional feature-map output |
| `train.py` | End-to-end training job on Modal GPU (dataset baked into the image) |
| `main.py` | Hosted Modal inference endpoint returning predictions + tensors for visualization |
| `cnn-visualizer/` | Next.js client to inspect inference outputs. See [its README](cnn-visualizer/README.md) |
| `voice_clips/` | A few sample clips for manual endpoint testing |

## Model

`AudioCNN` (in `model.py`) is a ResNet-style CNN operating on **128 × 87 mel-spectrogram inputs**:

- **Stem**: 7×7 conv (64 channels, stride 2) → BatchNorm → ReLU → max-pool
- **Residual stages**: 3, 4, 6, 3 blocks with 64 → 128 → 256 → 512 channels (first block of each stage downsamples with stride 2), each block `conv–bn–relu–conv–bn` + identity/projection shortcut
- **Head**: adaptive average pooling → dropout (0.5) → linear → `num_classes` (50 for ESC-50)

`forward(x, return_feature_maps=True)` additionally returns a dict of intermediate activations (stem, every residual block pre/post-add, each stage) — this is what the visualizer renders as heatmaps.

## Dataset and Training

`train.py` runs a Modal function (`app = modal.App("audio-cnn")`) on an **A10G** GPU with a 3-hour timeout:

- **Dataset**: [ESC-50](https://github.com/karolpiczak/ESC-50), downloaded and unpacked into `/opt/esc50-data` during the Modal image build
- **Split**: folds 1–4 train, fold 5 validation
- **Input transform**: mel-spectrogram at 22.05 kHz (`n_fft=1024`, `hop=512`, `n_mels=128`, `f_max=11025`) → amplitude-to-dB
- **Augmentation**:
  - SpecAugment-style masking (`FrequencyMasking(30)`, `TimeMasking(80)`)
  - mixup (`Beta(0.2, 0.2)` blending weights, applied with ~30% probability per batch, paired with the weighted mixup loss)
- **Optimization**: AdamW (`lr=5e-4`, `weight_decay=0.01`) + OneCycleLR (`max_lr=2e-3`, `pct_start=0.1`), label-smoothed cross-entropy (0.1), batch size 32, **100 epochs**
- **Artifacts** (Modal volume `esc50-modal`, mounted at `/model`):
  - `/model/best_model.pth` — best checkpoint (state dict, class list, accuracy, epoch)
  - `/model/tensorboard_logs/run_<timestamp>/` — TensorBoard training curves

## Inference API

`main.py` serves `app = modal.App("audio-cnn-inference")`: an `AudioClassifier` class on an **A10G** GPU that loads the checkpoint once at startup (`@modal.enter()`), scales down after **15 s** of inactivity, and exposes a FastAPI endpoint via `@modal.fastapi_endpoint`.

**Request** — `POST` with JSON body:

```json
{
  "audio_data": "<base64-encoded-wav-bytes>"
}
```

Audio is decoded with `soundfile`, mixed to mono if needed, and resampled to 44.1 kHz if it isn't already (the frontend uploads 44.1 kHz WAV).

**Response** — the current backend returns top-3 predictions, mean-aggregated activation heatmaps, the input spectrogram, and a downsampled waveform (≤ 8000 samples):

```json
{
  "predictions": [
    { "class": "helicopter", "confidence": 0.6 },
    { "class": "sneezing", "confidence": 0.05 },
    { "class": "dog", "confidence": 0.04 }
  ],
  "visulization": {
    "conv1": { "shape": [32, 22], "values": [[0.95, 0.84]] }
  },
  "input_spectogram": {
    "shape": [128, 87],
    "values": [[-31.2, -29.7]]
  },
  "waveform": {
    "values": [0.0, 0.01, -0.02],
    "sample_rate": 44100,
    "duration": 5.0
  }
}
```

> **Note:** two response keys are misspelled in the backend — `visulization` (should be `visualization`) and `input_spectogram` (should be `input_spectrogram`). The visualizer normalizes these to the canonical names internally.

## Getting Started

Prerequisites:

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) (or `pip`)
- Modal CLI authenticated (`modal token new`)
- Node.js 20+ / npm 11+ (only for the visualizer)

Install dependencies:

```bash
uv sync
```

### Train on Modal

```bash
modal run train.py
```

Downloads ESC-50 during the image build, trains on an A10G, and writes the best checkpoint to the `esc50-modal` volume.

### Serve / test inference

```bash
modal run main.py
```

Launches the inference endpoint (and, with the local entrypoint, sends `voice_clips/fl-mocking-birds-36124.mp3` through it and prints the top predictions). The visualizer's `cnn-visualizer/.env.example` shows the expected `NEXT_PUBLIC_INFERENCE_URL` — point it at your own endpoint URL printed by Modal.

### Run the visualizer frontend

```bash
cd cnn-visualizer
npm install
npm run dev
```

Open `http://localhost:3000`, upload a WAV, and inspect predictions, spectrogram, waveform, and layer activations.

## Notes

- Checkpoints, datasets, and TensorBoard logs live in Modal volumes / the image, not in git
- The deployed endpoint URL is generated by Modal from the app and class names
- This repo emphasizes practical experimentation and inspection over production hardening
