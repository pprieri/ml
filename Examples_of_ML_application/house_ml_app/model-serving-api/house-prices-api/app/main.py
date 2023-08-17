from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

from app.api import api_router
from app.config import settings, setup_app_logging

# setup logging as early as possible
setup_app_logging(config=settings)


app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

root_router = APIRouter()


@root_router.get("/")
def index(request: Request) -> Any:
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the House price prediction ML API</h1>"
        "<div>"
        "Check the docs and get a prediction: <a href='/docs'>here</a>"
        "</div>"
        "<div>"
        "The dataset comes from the following Kaggle competition: <a href='https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques' target='_blank'>House Prices - Advanced Regression Techniques</a>"
        "</div>"
        "<br> <!-- Line break for the blank line -->"
        "<div>"
        "This is just a sample of deploying ML model combining different tools."
        "</div>"
        "<div>"
        "This ML application could be used to be called via a front end and display the results in a UI, providing users with a user-friendly way to interact with the predictive capabilities. Alternatively, it could be seamlessly integrated and accessed from a mobile application, extending its reach to users on various devices and platforms."
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)


app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(root_router)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


if __name__ == "__main__":
    # Use this for debugging purposes only
    logger.warning("Running in development mode. Do not run like this in production.")
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001, log_level="debug")
