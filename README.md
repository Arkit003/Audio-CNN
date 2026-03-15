# Audio-CNN (ESC-50)

Audio classification project centered on a CNN pipeline for ESC-50 style environmental sound recognition, with Modal-based training/inference and a small visualization client.

## What This Repo Contains

- `train.py`: end-to-end training job on Modal GPU
- `model.py`: residual CNN architecture used for classification
- `main.py`: hosted inference endpoint that returns predictions + tensors for visualization
- `cnn-visualizer/`: lightweight Next.js app to inspect inference outputs

Frontend details: see `cnn-visualizer/README.md`.

This repo is primarily about the **audio CNN workflow** (data -> spectrogram -> model -> class probabilities + intermediate activations), not frontend engineering.

## Model Pipeline

1. Decode input audio and convert to mono if needed
2. Resample audio for model processing
3. Convert waveform to mel-spectrogram (`MelSpectrogram` + `AmplitudeToDB`)
4. Run residual CNN forward pass
5. Return top-k predictions and feature maps for interpretability

## Dataset and Training

- Dataset: ESC-50 (downloaded in Modal image build step)
- Split: folds 1-4 train, fold 5 validation
- Augmentations:
  - spectrogram masking (`FrequencyMasking`, `TimeMasking`)
  - mixup on minibatches
- Optimization:
  - `AdamW`
  - `OneCycleLR`
  - label smoothing cross-entropy
- Artifact: best checkpoint saved to Modal volume as `/model/best_model.pth`

## Inference Output Contract

Inference accepts:

```json
{
  "audio_data": "<base64-encoded-wav-bytes>"
}
```

Inference returns:

- `predictions`: top classes with confidence
- `visulization` (legacy key in current backend): layer activation maps
- `input_spectogram` (legacy key in current backend): model input spectrogram
- `waveform`: downsampled waveform values, sample rate, duration

Note: the visualizer normalizes legacy misspelled keys (`visulization`, `input_spectogram`) to canonical names.

## Local Setup

Prerequisites:

- Python 3.11+
- `uv` (recommended) or `pip`
- Modal CLI authenticated (`modal token new`)

Install dependencies:

```bash
uv sync
```

## Run

### Launch inference endpoint on Modal

```bash
modal run main.py
```

### Launch training job on Modal

```bash
modal run train.py
```

### (Optional) Run visualizer frontend

```bash
cd cnn-visualizer
npm install
npm run dev
```

## Notes

- Checkpoints and logs are stored in Modal volumes, not committed to git
- This repo currently emphasizes practical experimentation and inspection over production hardening
