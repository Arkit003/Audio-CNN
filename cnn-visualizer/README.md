# Audio CNN Inference Visualizer

Visualization client for an ESC-50 style audio classification CNN.

## What This Project Is

This repository is mainly about **inspecting Audio CNN behavior** at inference time.

You upload a `.wav`, the backend CNN runs inference, and this app renders model outputs so you can see both the prediction and internal activations:

- class probabilities (top-k predictions)
- input spectrogram used by the model
- raw waveform for quick sanity checking
- convolutional feature maps and internal layer activations

The frontend is intentionally lightweight (vibe-coded) and acts as a thin viewer for CNN outputs.

## Audio CNN Focus

- **Task**: environmental sound classification (ESC-50 label set)
- **Input**: WAV audio encoded as base64 in JSON
- **Model-side outputs exposed**:
  - predicted classes with confidence scores
  - input spectrogram tensor
  - per-layer convolutional/intermediate tensors for interpretability
- **Goal**: fast qualitative debugging of what the audio CNN is attending to

## UI/Tooling Stack

- Next.js 15 (App Router)
- React 19
- TypeScript
- Tailwind CSS 4
- shadcn-style UI primitives

## Getting Started

### Prerequisites

- Node.js 20+
- npm 11+

### Install

```bash
npm install
```

### Run in development

```bash
npm run dev
```

Open `http://localhost:3000` and upload a WAV file.

## Available Scripts

- `npm run dev` - start local dev server
- `npm run build` - create production build
- `npm run start` - run production server
- `npm run lint` - run ESLint
- `npm run typecheck` - run TypeScript checks
- `npm run check` - lint + typecheck
- `npm run format:check` - check Prettier formatting
- `npm run format:write` - apply Prettier formatting

## Inference API Contract (Model Output)

Request payload sent to the inference service:

```json
{
  "audio_data": "<base64-wav-bytes>"
}
```

Normalized response shape consumed by this viewer:

```json
{
  "predictions": [{ "class": "dog", "confidence": 0.91 }],
  "visualization": { "conv1": { "shape": [32, 22], "values": [[0.1]] } },
  "input_spectrogram": { "shape": [128, 128], "values": [[0.2]] },
  "waveform": { "values": [0.0, 0.1], "sample_rate": 16000, "duration": 1.0 }
}
```

Note: the frontend currently includes compatibility normalization for legacy misspellings returned by some backend versions (`visulization`, `input_spectogram`).

## Project Structure

- `src/app/page.tsx` - upload + inference request + response normalization
- `src/components/FeatureMap.tsx` - feature-map tensor rendering
- `src/components/Waveform.tsx` - waveform renderer
- `src/components/ColorScale.tsx` - activation color legend
- `src/components/ui/*` - minimal UI shell

## Notes

- Input files should be WAV (`.wav`)
- Large files may increase inference latency
- This repo does not train the CNN; it visualizes outputs from the hosted inference endpoint
