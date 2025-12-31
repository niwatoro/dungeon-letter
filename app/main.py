from __future__ import annotations

import random
from typing import Iterable, Sequence

from flask import Flask, render_template, request

from dungeon import (
    DEFAULT_FLOOR_COLOR,
    DEFAULT_MASK,
    DEFAULT_MASK_HEIGHT,
    DEFAULT_MASK_WIDTH,
    DEFAULT_MESSAGE_WALL_COLOR,
    DEFAULT_WALL_COLOR,
    MAX_MASK_HEIGHT,
    MAX_MASK_WIDTH,
    MESSAGE_MIN_COL,
    MESSAGE_START_COL,
    MESSAGE_START_ROW,
    MIN_MASK_HEIGHT,
    MIN_MASK_WIDTH,
    RenderOptions,
    encode_png,
    render_message_maze,
)

app = Flask(__name__, template_folder="../templates", static_folder="../static")

DEFAULT_MESSAGE = "HAPPY\n NEW\n YEAR"


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


def _parse_mask_override(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    lines = value.splitlines()
    if not any(line.strip() for line in lines):
        return None
    width = max(len(line) for line in lines)
    height = len(lines)
    if width < MIN_MASK_WIDTH or width > MAX_MASK_WIDTH:
        raise ValueError(
            f"Mask width must be between {MIN_MASK_WIDTH} and {MAX_MASK_WIDTH} columns."
        )
    if height < MIN_MASK_HEIGHT or height > MAX_MASK_HEIGHT:
        raise ValueError(
            f"Mask height must be between {MIN_MASK_HEIGHT} and {MAX_MASK_HEIGHT} rows."
        )
    allowed_chars = {"#", ".", " "}
    allowed_chars.update("0123456789abcdefABCDEF")
    for line_idx, line in enumerate(lines, start=1):
        for col_idx, ch in enumerate(line, start=1):
            if ch not in allowed_chars:
                raise ValueError(
                    f"Mask contains an unsupported character '{ch}' at row {line_idx}, column {col_idx}."
                )
    return lines


def _build_options(form: dict) -> RenderOptions:
    message_value = form.get("message")
    message = DEFAULT_MESSAGE if message_value is None else message_value
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
    message_wall_color = _parse_color(
        form.get("message_wall_color"),
        DEFAULT_MESSAGE_WALL_COLOR,
        label="Message wall color",
    )
    floor_color = _parse_color(
        form.get("floor_color"), DEFAULT_FLOOR_COLOR, label="Floor color"
    )
    mask_override = _parse_mask_override(form.get("mask"))
    effective_mask_height = len(mask_override) if mask_override else mask_height
    effective_mask_width = (
        max(len(line) for line in mask_override) if mask_override else mask_width
    )
    message_start_row = _clamp_int(
        form.get("message_start_row"),
        MESSAGE_START_ROW,
        minimum=0,
        maximum=max(0, effective_mask_height - 3),
    )
    message_start_col = _clamp_int(
        form.get("message_start_col"),
        MESSAGE_START_COL,
        minimum=MESSAGE_MIN_COL,
        maximum=max(MESSAGE_MIN_COL, effective_mask_width - 3),
    )
    return RenderOptions(
        message=message,
        seed=seed,
        mask_width=mask_width,
        mask_height=mask_height,
        mask_override=mask_override,
        wall_color=wall_color,
        message_wall_color=message_wall_color,
        floor_color=floor_color,
        scale=scale,
        dpi=dpi,
        message_start_row=message_start_row,
        message_start_col=message_start_col,
    )


def _stats(raw: Sequence[Sequence[int]]):
    height = len(raw)
    if height == 0 or not raw[0]:
        return {"width": 0, "height": 0, "floor_pct": 0.0}
    width = len(raw[0])
    total = width * height
    floor_count = sum(1 for row in raw for value in row if value == 0)
    floor_ratio = floor_count / total if total else 0
    return {
        "width": width,
        "height": height,
        "floor_pct": round(floor_ratio * 100, 2),
    }


@app.route("/", methods=["GET", "POST"])
async def index() -> str:
    error = None
    image_data = None
    stats = None
    mask_lines = None
    options = RenderOptions(message=DEFAULT_MESSAGE)
    mask_input_value = "\n".join(DEFAULT_MASK)
    mask_actual_width = options.mask_width
    mask_actual_height = options.mask_height

    if request.method == "POST":
        try:
            options = _build_options(request.form)
            image_pixels, raw, mask, used_rows, _ = render_message_maze(options)
            image_data = f"data:image/png;base64,{encode_png(image_pixels, options.dpi)}"
            stats = _stats(raw)
            mask_lines = [
                {"index": row_idx, "text": mask[row_idx]}
                for pair in used_rows
                for row_idx in pair
            ]
            mask_actual_width = len(mask[0]) if mask else options.mask_width
            mask_actual_height = len(mask)
        except Exception as exc:  # pylint: disable=broad-except
            error = str(exc)

    message_width = max(0, mask_actual_width - 2 - options.message_start_col)
    message_row_max = max(0, mask_actual_height - 3)
    message_col_max = max(MESSAGE_MIN_COL, mask_actual_width - 3)
    context = {
        "image_data": image_data,
        "stats": stats,
        "error": error,
        "options": options.to_dict(),
        "message_width": message_width,
        "mask_preview": mask_lines,
        "mask_min_width": MIN_MASK_WIDTH,
        "mask_max_width": MAX_MASK_WIDTH,
        "mask_min_height": MIN_MASK_HEIGHT,
        "mask_max_height": MAX_MASK_HEIGHT,
        "mask_text": mask_input_value,
        "mask_actual_width": mask_actual_width,
        "mask_actual_height": mask_actual_height,
        "wall_color_hex": _rgb_to_hex(options.wall_color),
        "message_wall_color_hex": _rgb_to_hex(options.message_wall_color),
        "floor_color_hex": _rgb_to_hex(options.floor_color),
        "message_row_max": message_row_max,
        "message_col_min": MESSAGE_MIN_COL,
        "message_col_max": message_col_max,
    }
    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(port=8000)
