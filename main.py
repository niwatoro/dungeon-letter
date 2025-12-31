from __future__ import annotations

import base64
import io
import os
import random
from typing import Iterable

import numpy as np
from flask import Flask, render_template, request

from dungeon_drawer import (
    DEFAULT_FLOOR_COLOR,
    DEFAULT_MASK_HEIGHT,
    DEFAULT_MASK_WIDTH,
    DEFAULT_WALL_COLOR,
    MAX_MASK_HEIGHT,
    MAX_MASK_WIDTH,
    MESSAGE_START_COL,
    MIN_MASK_HEIGHT,
    MIN_MASK_WIDTH,
    RenderOptions,
    render_message_maze,
)

app = Flask(__name__)

DEFAULT_MESSAGE = "DUNGEON"


def _encode_png(img, dpi: int) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=(dpi, dpi))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _clamp_int(value: str | None, default: int, *, minimum: int, maximum: int) -> int:
    if not value:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _parse_seed(seed_raw: str | None) -> int:
    seed_str = (seed_raw or "").strip()
    if not seed_str:
        return random.randint(0, 999999)
    try:
        return int(seed_str)
    except ValueError as exc:
        raise ValueError("Maze seed must be an integer.") from exc


def _parse_color(
    value: str | None, default: tuple[int, int, int], label: str
) -> tuple[int, int, int]:
    text = (value or "").strip()
    if not text:
        return default
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        raise ValueError(f"{label} must be a #RRGGBB value.")
    try:
        r = int(text[0:2], 16)
        g = int(text[2:4], 16)
        b = int(text[4:6], 16)
        return (r, g, b)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{label} must be a #RRGGBB value.") from exc


def _rgb_to_hex(rgb: Iterable[int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_options(form: dict) -> RenderOptions:
    message = form.get("message", DEFAULT_MESSAGE).strip() or DEFAULT_MESSAGE
    seed = _parse_seed(form.get("seed"))
    scale = _clamp_int(form.get("scale"), 12, minimum=1, maximum=40)
    dpi = _clamp_int(form.get("dpi"), 500, minimum=72, maximum=1200)
    mask_width = _clamp_int(
        form.get("mask_width"),
        DEFAULT_MASK_WIDTH,
        minimum=MIN_MASK_WIDTH,
        maximum=MAX_MASK_WIDTH,
    )
    mask_height = _clamp_int(
        form.get("mask_height"),
        DEFAULT_MASK_HEIGHT,
        minimum=MIN_MASK_HEIGHT,
        maximum=MAX_MASK_HEIGHT,
    )
    wall_color = _parse_color(
        form.get("wall_color"), DEFAULT_WALL_COLOR, label="Wall color"
    )
    floor_color = _parse_color(
        form.get("floor_color"), DEFAULT_FLOOR_COLOR, label="Floor color"
    )
    return RenderOptions(
        message=message,
        seed=seed,
        mask_width=mask_width,
        mask_height=mask_height,
        wall_color=wall_color,
        floor_color=floor_color,
        scale=scale,
        dpi=dpi,
    )


def _stats(raw: np.ndarray):
    floor_ratio = np.mean(raw == 0)
    return {
        "width": int(raw.shape[1]),
        "height": int(raw.shape[0]),
        "floor_pct": round(floor_ratio * 100, 2),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    image_data = None
    stats = None
    mask_lines = None
    options = RenderOptions(message=DEFAULT_MESSAGE)

    if request.method == "POST":
        try:
            options = _build_options(request.form)
            img, raw, mask, used_rows = render_message_maze(options)
            image_data = f"data:image/png;base64,{_encode_png(img, options.dpi)}"
            stats = _stats(raw)
            mask_lines = [
                {"index": row_idx, "text": mask[row_idx]}
                for pair in used_rows
                for row_idx in pair
            ]
        except Exception as exc:  # pylint: disable=broad-except
            error = str(exc)

    message_width = max(0, options.mask_width - 2 - MESSAGE_START_COL)
    return render_template(
        "index.html",
        image_data=image_data,
        stats=stats,
        error=error,
        options=options,
        message_width=message_width,
        mask_preview=mask_lines,
        mask_min_width=MIN_MASK_WIDTH,
        mask_max_width=MAX_MASK_WIDTH,
        mask_min_height=MIN_MASK_HEIGHT,
        mask_max_height=MAX_MASK_HEIGHT,
        wall_color_hex=_rgb_to_hex(options.wall_color),
        floor_color_hex=_rgb_to_hex(options.floor_color),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=True, host="0.0.0.0", port=port)
