"""
Input validation shared across projects -- file type/size checks
before anything touches the model.
"""
from fastapi import UploadFile, HTTPException
from app.config import settings

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def validate_image(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
