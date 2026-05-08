from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from PIL import Image


class InsightFaceBoxDetector:
    """Detect normalized face boxes with InsightFace.

    Returned boxes use the ExpressionAdapter format:
    `[x1, y1, x2, y2]` normalized to `[0, 1]`.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: list[str] | None = None,
        ctx_id: int = 0,
        det_size: tuple[int, int] = (640, 640),
        min_score: float = 0.0,
        margin: float = 0.0,
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "InsightFaceBoxDetector requires `insightface`. "
                "Install it with `pip install insightface`."
            ) from exc

        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.min_score = min_score
        self.margin = margin

    def detect_image(self, image: Image.Image) -> torch.Tensor | None:
        """Return the largest detected face box, or `None` if no face is found."""

        width, height = image.size
        frame = _pil_to_bgr(image)
        faces = [
            face
            for face in self.app.get(frame)
            if float(getattr(face, "det_score", 1.0)) >= self.min_score
        ]
        if not faces:
            return None

        face = max(faces, key=lambda item: _bbox_area(item.bbox))
        return _normalize_bbox(face.bbox, width=width, height=height, margin=self.margin)

    def detect_frames(
        self,
        frames: Iterable[Image.Image],
        fallback_box: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one normalized face box per frame with shape `[F, 4]`.

        If a frame has no detected face, `fallback_box` is used when provided.
        Otherwise the previous detected box is reused. If the first frame has no
        face and no fallback exists, a `ValueError` is raised.
        """

        boxes = []
        previous_box = fallback_box

        for frame in frames:
            box = self.detect_image(frame)
            if box is None:
                box = previous_box
            if box is None:
                raise ValueError("No face detected and no fallback_box was provided.")

            boxes.append(box)
            previous_box = box

        if not boxes:
            raise ValueError("detect_frames requires at least one frame.")

        return torch.stack(boxes, dim=0)


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def _bbox_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox
    return max(float(x2 - x1), 0.0) * max(float(y2 - y1), 0.0)


def _normalize_bbox(
    bbox: np.ndarray,
    width: int,
    height: int,
    margin: float,
) -> torch.Tensor:
    x1, y1, x2, y2 = [float(value) for value in bbox]

    if margin > 0.0:
        box_width = x2 - x1
        box_height = y2 - y1
        x1 -= box_width * margin
        x2 += box_width * margin
        y1 -= box_height * margin
        y2 += box_height * margin

    box = torch.tensor(
        [x1 / width, y1 / height, x2 / width, y2 / height],
        dtype=torch.float32,
    )
    return box.clamp(0.0, 1.0)
