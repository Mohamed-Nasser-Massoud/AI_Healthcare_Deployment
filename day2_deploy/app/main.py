import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import model_loader
from app.routers import cv, dl, ml, rag
from app.schemas import HealthResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once, when the container starts -- not per-request. This is the
    # one thing to get right in any deployed ML API: load everything here,
    # never inside a route handler.
    model_loader.load_all()
    yield


app = FastAPI(
    title="RespiraAI",
    description=(
        "Deployment capstone: classical ML + deep learning (lung cancer risk), "
        "computer vision (chest X-ray pneumonia detection), and a RAG pipeline "
        "(respiratory conditions Q&A) behind one API."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Wide open for a weekend build the frontend is served from the same origin
# as the API, so this isn't even load-bearing right now -- but any real
# deployment tends to grow a second origin eventually (a separate static
# host, a mobile app, etc.), and that's the moment a locked-down CORS policy
# actually matters. Narrow allow_origins to your real frontend's origin(s)
# before this goes anywhere beyond localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml.router)
app.include_router(dl.router)
app.include_router(cv.router)
app.include_router(rag.router)


@app.get("/health", response_model=HealthResponse)
def health():
    return model_loader.health_status()


# Registered LAST and mounted at "/" on purpose: FastAPI matches routes in
# registration order, so the specific API routes above always win for their
# exact paths, and this catch-all only serves everything else -- "/" itself,
# plus any other static asset under frontend/. Swap the order and the
# static mount would shadow every API route with a 404.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
