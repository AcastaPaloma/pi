"""Local, read-only dataset replay API. Open http://127.0.0.1:8000/docs."""
from __future__ import annotations

from io import BytesIO
from typing import Annotated, Literal

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
import numpy as np
from PIL import Image

from .dataset import CAMERAS, REPO, REVISION, SUBSET, FoldingDataset


def create_app(dataset: FoldingDataset | None = None) -> FastAPI:
    ds = dataset if dataset is not None else FoldingDataset()
    app = FastAPI(title="Folding encoder data", version="1.0.0",
                  description="Read-only LeHome replay. Future subgoals and actions are training labels, not live robot observations.")

    @app.exception_handler(KeyError)
    @app.exception_handler(IndexError)
    async def not_found(request: Request, exc: Exception):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def missing_file(request: Request, exc: Exception):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    def cameras(episode: int, frame: int, size: int = 448) -> dict[str, str]:
        return {cam: f"/v1/episodes/{episode}/frames/{frame}/cameras/{cam}.png?size={size}" for cam in CAMERAS}

    def with_images(sample: dict) -> dict:
        ep = sample["episode"]
        sample["cameras"] = cameras(ep, sample["frame"])
        sample["history"]["cameras"] = [cameras(ep, frame) for frame in sample["history"]["frames"]]
        sample["visual_subgoal"]["cameras"] = cameras(ep, sample["visual_subgoal"]["frame"])
        return sample

    @app.get("/v1/dataset")
    def dataset_info():
        return {"repo": REPO, "revision": REVISION, "subset": SUBSET, "fps": ds.fps,
                "episodes": len(ds.episodes), "frames": sum(ep["length"] for ep in ds.episodes.values()),
                "camera_order": CAMERAS, "features": ds.info["features"],
                "splits": {name: len(ds.episode_ids(name)) for name in ("train", "validation")},
                "split_policy": "lehome-v1 SHA256 rank, 10% held-out episodes",
                "missing_annotations": ["subtask", "subtask_boundaries", "quality", "mistake", "speed"],
                "mode": "dataset_replay"}

    @app.get("/v1/episodes")
    def episodes(split: Literal["train", "validation"] | None = None):
        return [{"episode": ep, "length": ds.episode(ep)["length"], "split": ds.split(ep)} for ep in ds.episode_ids(split)]

    @app.get("/v1/normalization")
    def normalization():
        return ds.train_statistics()

    @app.get("/v1/episodes/{episode}/frames/{frame}/sample")
    def sample(episode: int, frame: int,
               horizon: Annotated[int, Query(ge=1, le=200)] = 50,
               history: Annotated[int, Query(ge=1, le=6)] = 1,
               goal_seconds: Annotated[float | None, Query(ge=0, le=4)] = None,
               seed: Annotated[int, Query(ge=0)] = 0):
        return with_images(ds.sample(episode, frame, horizon=horizon, history=history, goal_seconds=goal_seconds, seed=seed))

    @app.get("/v1/episodes/{episode}/frames/{frame}/observations")
    def observations(episode: int, frame: int):
        s = ds.sample(episode, frame)
        return {"episode": episode, "frame": frame, "timestamp_seconds": s["timestamp_seconds"],
                "camera_order": CAMERAS, "cameras": cameras(episode, frame), "proprioception": s["proprioception"]}

    @app.get("/v1/episodes/{episode}/frames/{frame}/proprioception")
    def proprioception(episode: int, frame: int):
        return ds.sample(episode, frame)["proprioception"]

    @app.get("/v1/episodes/{episode}/frames/{frame}/text")
    def text(episode: int, frame: int):
        return ds.sample(episode, frame)["text"]

    @app.get("/v1/episodes/{episode}/frames/{frame}/actions")
    def actions(episode: int, frame: int, horizon: Annotated[int, Query(ge=1, le=200)] = 50):
        return ds.sample(episode, frame, horizon=horizon)["actions"]

    @app.get("/v1/episodes/{episode}/frames/{frame}/subgoal")
    def subgoal(episode: int, frame: int,
                goal_seconds: Annotated[float | None, Query(ge=0, le=4)] = None,
                seed: Annotated[int, Query(ge=0)] = 0):
        return with_images(ds.sample(episode, frame, goal_seconds=goal_seconds, seed=seed))["visual_subgoal"]

    @app.get("/v1/episodes/{episode}/frames/{frame}/cameras/{camera}.png", response_class=Response)
    def camera_image(episode: int, frame: int, camera: str, size: Annotated[int, Query(ge=16, le=1024)] = 448):
        output = BytesIO()
        Image.fromarray(ds.image(episode, frame, camera, size)).save(output, format="PNG")
        return Response(output.getvalue(), media_type="image/png")

    @app.get("/v1/episodes/{episode}/frames/{frame}/sample.npz", response_class=Response)
    def arrays(episode: int, frame: int,
               size: Annotated[int, Query(ge=16, le=1024)] = 448,
               horizon: Annotated[int, Query(ge=1, le=200)] = 50,
               history: Annotated[int, Query(ge=1, le=6)] = 1,
               goal_seconds: Annotated[float | None, Query(ge=0, le=4)] = None,
               seed: Annotated[int, Query(ge=0)] = 0):
        output = BytesIO()
        np.savez_compressed(output, **ds.arrays(episode, frame, size=size, horizon=horizon,
                                             history=history, goal_seconds=goal_seconds, seed=seed))
        return Response(output.getvalue(), media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="episode-{episode}-frame-{frame}.npz"'})

    return app
