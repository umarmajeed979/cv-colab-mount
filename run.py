"""Local dev entry point: python run.py"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:create_app", factory=True, host="0.0.0.0", port=8000, reload=True)
