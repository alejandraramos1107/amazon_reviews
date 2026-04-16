"""
Simple FastAPI interface for Amazon reviews authorship predictions.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.serving.model_loader import model_loader


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Amazon Reviews Authorship Demo",
    description="Simple interface to inspect predictions from the best MLflow model.",
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Loading best model from MLflow...")
    model_loader.load()
    logger.info("Interface ready.")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    loaded = model_loader.load()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": None,
            "max_index": len(loaded.dataset) - 1,
            "model_name": loaded.model_name,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, row_index: int = Form(...)) -> HTMLResponse:
    loaded = model_loader.load()
    try:
        result = model_loader.predict_by_index(row_index)
    except IndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": result,
            "max_index": len(loaded.dataset) - 1,
            "model_name": loaded.model_name,
        },
    )


@app.get("/health")
async def health() -> dict:
    loaded = model_loader.load()
    return {
        "status": "healthy",
        "model_name": loaded.model_name,
        "rows_available": len(loaded.dataset),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
