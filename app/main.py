from fastapi import FastAPI
import uvicorn
from .routes import router
from .websocket_routes import router as ws_router

app = FastAPI(
    title="MaZe",
    description="MaZe is a web application that helps you find the best place to live based on your needs.",
    version="0.0.1",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(router)
app.include_router(ws_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)