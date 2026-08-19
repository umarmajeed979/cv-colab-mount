"""
Project-specific middleware (request logging, size limits, etc).
"""
import time
from fastapi import Request
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)")
    return response
