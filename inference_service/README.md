# Dermalyze Inference Service

Standalone inference API for frontend deployment. This package is intentionally decoupled from training code in `skin_lesion_classifier/`.

## What To Deploy

- `inference_service/app.py`
- `inference_service/predictor.py`
- `inference_service/models/`
- `inference_service/metadata.py`
- `inference_service/tta_constants.py`
- Your model checkpoint at `inference_service/model/checkpoint_best.pt` (or set `MODEL_CHECKPOINT`)

## Install

```bash
cd inference_service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run API

```bash
uvicorn inference_service.app:app --host 0.0.0.0 --port 8000
```

Alternative (from inside `inference_service/`):

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Environment Variables

- `MODEL_CHECKPOINT` (default: `inference_service/model/checkpoint_best.pt`)
- `MODEL_IMAGE_SIZE` (default: `224`)
- `USE_TTA` (`true`/`false`, default: `false`)
- `TTA_MODE` (`light` | `medium` | `full`, default: `medium`)
- `TTA_AGGREGATION` (`mean` | `geometric_mean` | `max`, default: `geometric_mean`)
- `CORS_ORIGINS` (comma-separated frontend origins)

## Frontend Contract

- `POST /classify`
- Content type: `multipart/form-data`
- File field name: `file`
- Accepted image types: JPEG, PNG, WebP
- Max upload size: 20 MB

Response:

```json
{
  "classes": [
    {"id": "mel", "name": "Melanoma", "score": 67.4}
  ]
}
```

## Frontend Env Example

```env
VITE_API_URL=http://localhost:8000
```
