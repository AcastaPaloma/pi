"""Small batch contact sheet for notebooks and saved previews."""
import math
import numpy as np
from PIL import Image, ImageDraw


def observation_grid(batch, *, camera=None, columns=16, tile_size=128):
    """Return a PIL image. In a notebook: display(observation_grid(batch))."""
    images = batch["images"]
    cameras = list(batch["camera_names"])
    if images.ndim != 5 or images.shape[1] != len(cameras) or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError("Expected RGB uint8 images [B,C,H,W,3]")
    if len(images) < 1 or not isinstance(columns, int) or columns < 1 or not 16 <= tile_size <= 1024:
        raise ValueError("Expected a nonempty batch, positive columns, and tile_size 16..1024")
    camera = camera if camera is not None else cameras[-1]
    if camera not in cameras:
        raise ValueError(f"Unknown camera {camera}; choose {cameras}")
    column_count = min(columns, len(images))
    canvas = Image.new("RGB", (column_count * tile_size, math.ceil(len(images) / column_count) * (tile_size + 20)), "#151922")
    draw = ImageDraw.Draw(canvas)
    for i, frame in enumerate(images[:, cameras.index(camera)]):
        x, y = (i % column_count) * tile_size, (i // column_count) * (tile_size + 20)
        canvas.paste(Image.fromarray(frame).resize((tile_size, tile_size), Image.Resampling.BILINEAR), (x, y + 20))
        label = f"#{i}"
        if "episode" in batch:
            label += f" ep{batch['episode'][i]} f{batch['frame'][i]}"
        draw.text((x + 4, y + 4), label, fill="white")
    return canvas
