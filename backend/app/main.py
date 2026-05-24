from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles

from app.api.routes.pose import router as pose_router
from app.api.routes.calibration import router as calibration_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load persisted calibration params on startup
    import json
    from pathlib import Path
    params_path = Path(__file__).parent / "core" / "calibration_params.json"
    if params_path.exists():
        try:
            data = json.loads(params_path.read_text())
            from app.api.routes import calibration
            calibration._params["scale_x"] = data.get("scale_x", 2.5)
            calibration._params["scale_y"] = data.get("scale_y", 3.5)
            calibration._params["scale_z"] = data.get("scale_z", 2.5)
        except Exception:
            pass
    yield


app = FastAPI(
    title="My Digital Twin",
    description="实时人体姿态识别后端服务",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,   # overridden below with local assets
    redoc_url=None,  # disabled (uses CDN)
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pose_router)
app.include_router(calibration_router)


@app.get("/docs", include_in_schema=False)
async def swagger_docs():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url="/static/favicon.png",
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
