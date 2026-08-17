"""Production entry point for platforms expecting a WSGI/ASGI callable."""
from app import create_app

app = create_app()
