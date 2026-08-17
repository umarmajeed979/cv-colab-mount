"""
Route definitions only. No business logic here -- every endpoint
delegates to core/predictor.py and returns via schemas.py.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.api.schemas import PredictResponse, HealthResponse
from app.core.predictor import Predictor
from app.utils.validators import validate_image
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
predictor = Predictor()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=predictor.is_loaded())


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    validate_image(file)
    try:
        image_bytes = await file.read()
        result = predictor.predict(image_bytes)
        return result
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
