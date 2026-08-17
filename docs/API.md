# API

Base URL: `http://localhost:8000/api/v1`

## GET /health
Returns `{"status": "ok", "model_loaded": bool}`.

## POST /predict
Multipart form upload, field name `file` (jpeg/png/webp, max 10MB).

Response:
```json
{
  "overlay_image_base64": "<png bytes, base64>",
  "mask_image_base64": "<png bytes, base64>",
  "class_coverage": [
    {"label": "road", "pixel_percentage": 34.2},
    {"label": "car", "pixel_percentage": 8.7}
  ],
  "inference_time_ms": 142.3,
  "model_version": "0.1.0"
}
```
`class_coverage` is sorted descending by `pixel_percentage` and only
includes classes actually present in the predicted mask.
