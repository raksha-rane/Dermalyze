# Dermalyze Frontend

React + Vite + TypeScript frontend for authenticated dermoscopic image upload, AI classification, and longitudinal result tracking.

>  DISCLAIMER: Educational/research purposes only. Not for medical diagnosis.

## Overview

The frontend provides:

- Supabase authentication flows (login, signup, password reset, email verification)
- Image upload and classification orchestration against the inference API
- Results visualization across 7 lesion classes, including model uncertainty scores and calibrated confidence
- Trust Layer integration: surfacing image quality flags (blur, exposure) and safe abstention routing
- User-specific analysis history and trends dashboards

Class IDs used throughout the UI and backend:

- `akiec` - Actinic keratoses / intraepithelial carcinoma
- `bcc` - Basal cell carcinoma
- `bkl` - Benign keratosis-like lesions
- `df` - Dermatofibroma
- `mel` - Melanoma
- `nv` - Melanocytic nevi
- `vasc` - Vascular lesions

## Tech Stack

- React 19
- TypeScript
- Vite 6
- Tailwind CSS
- Supabase JS SDK

## Live Deployment

The backend inference API is deployed and accessible at:
- **API URL**: `https://asmit404-dermalyze.hf.space/`
- **Hugging Face Space**: [asmit404/dermalyze](https://huggingface.co/spaces/asmit404/dermalyze)

You can configure the frontend to use this deployed API by setting `VITE_API_URL=https://asmit404-dermalyze.hf.space` in your `.env.local` file.

## Prerequisites

1. Node.js 18+.
2. Running inference API (see `../inference_service/README.md`).
3. Supabase project for auth + data persistence.

## Setup

```bash
cd frontend
npm install
```

Create `frontend/.env.local` with the variables below.

## Environment Variables

Required:

- `VITE_SUPABASE_URL` - Supabase project URL.
- `VITE_SUPABASE_ANON_KEY` - Supabase anonymous key.

Optional:

- `VITE_API_URL` - Absolute backend URL. If omitted, frontend uses `/api`.

Dev-only override:

- `BACKEND_URL` - Overrides Vite proxy target for `/api` (default: `http://localhost:8000`).

Example `frontend/.env.local`:

```env
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
VITE_API_URL=http://localhost:8000
```

## Development

```bash
npm run dev
```

Vite is configured to run on `http://localhost:3000`.

If `VITE_API_URL` is not set:

- frontend requests `POST /api/classify`
- Vite proxy forwards `/api/*` to `http://localhost:8000` by default
- override target with:

```bash
BACKEND_URL=http://localhost:9000 npm run dev
```

## Scripts

- `npm run dev` - start dev server
- `npm run build` - build production bundle
- `npm run preview` - preview production build
- `npm run lint` - run ESLint
- `npm run format` - run Prettier

## Runtime Behavior Notes

- API requests include Supabase Bearer tokens.
- The classify call refreshes tokens if expiry is within 60 seconds.
- Supabase auth state is stored in `window.sessionStorage` (session ends on browser/tab close).
- Idle session guard: warning around 28 minutes inactivity, sign-out around 30 minutes inactivity on protected routes.
- Result exports and certain UI features may be restricted based on image quality flags (e.g., blur, exposure) and model uncertainty.
- Analysis images are encrypted client-side before upload. The encryption key is stored locally on the client device, so encrypted images can only be decrypted on devices that hold the same key.

## Backend Contract

Classification request:

- method: `POST`
- path: `/classify` (or `/api/classify` when proxying)
- body: `multipart/form-data` with `file` field
- auth: `Authorization: Bearer <supabase_access_token>`

Response shape:

```json
{
  "classes": [
    { "id": "mel", "name": "Melanoma", "score": 87.42 }
  ],
  "prediction": "mel",
  "calibrated_confidence": 0.8742,
  "uncertainty": {
    "score": 0.23,
    "normalized_entropy": 0.35,
    "top2_margin": 0.7811,
    "variation_ratio": 0.1258
  },
  "quality_flags": [],
  "recommendation": "classify"
}
```

`classes` are sorted descending by `score` before rendering.

## Supabase Setup

Use `frontend/supabase_setup.sql` to provision:

- `public.analyses` table
- row-level security policies
- storage policies for analysis images
- `get_dashboard_stats()` RPC function

## Related Modules

- Inference API: [`../inference_service/README.md`](../inference_service/README.md)
- Training pipeline: [`../skin_lesion_classifier/README.md`](../skin_lesion_classifier/README.md)



