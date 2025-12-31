import base64
import io
import random
from collections import deque
from dataclasses import dataclass
from typing import Sequence, TypeVar

from PIL import Image

U, D, L, R = 8, 4, 2, 1  # up, down, left, right

MESSAGE_START_ROW = 3  # 4th row from the top
MESSAGE_LINE_GAP = 1
MESSAGE_START_COL = 3  # 4th column from the left (0-indexed)
MESSAGE_MIN_COL = 1

DEFAULT_MASK = [
    "#.#############",
    "#.............#",
    "#.............#",
    "#.............#",
    "#.............#",
    "#.............#",
    "#.............#",
    "#.............#",
    "#.............#",
    "#.............#",
    "#.............#",
    "#.............#",
    "#.............#",
    "#############.#",
]

DEFAULT_MASK_HEIGHT = len(DEFAULT_MASK)
DEFAULT_MASK_WIDTH = len(DEFAULT_MASK[0])

MESSAGE_MAX_COLUMNS = (
    len(DEFAULT_MASK[0]) - 2
) - MESSAGE_START_COL  # usable columns to the right

MIN_MASK_HEIGHT = max(MESSAGE_START_ROW + 3, 6)
MIN_MASK_WIDTH = max(MESSAGE_START_COL + 2, 5)
MAX_MASK_HEIGHT = 80
MAX_MASK_WIDTH = 80

DIGIT_CODES = {
    "0": ("4", "8"),
    "1": ("e", "e"),
    "2": ("2", "1"),
    "3": ("2", "2"),
    "4": ("8", "6"),
    "5": ("1", "2"),
    "6": ("1", "0"),
    "7": ("6", "e"),
    "8": ("0", "0"),
    "9": ("0", "2"),
}

_LETTER_CODES_BASE = {
    "A": ("0", "4"),
    "B": ("9", "0"),
    "C": ("5", "9"),
    "D": ("a", "0"),
    "E": ("1", "1"),
    "F": ("1", "5"),
    "G": ("5", "8"),
    "H": ("8", "4"),
    "I": ("d", "d"),
    "J": ("e", "8"),
    "K": ("1", "4"),
    "L": ("d", "9"),
    "M": ("44", "cc"),
    "N": ("4", "c"),
    "O": (".", "0"),
    "P": ("0", "5"),
    "Q": ("0", "6"),
    "R": (".", "5"),
    "S": ("9", "2"),
    "T": ("9", "1"),
    "U": ("c", "8"),
    "V": (".", "8"),
    "W": ("cc", "88"),
    "X": ("8", "4"),
    "Y": ("8", "2"),
    "Z": ("6", "9"),
}

LETTER_CODES = {}
for key, value in _LETTER_CODES_BASE.items():
    LETTER_CODES[key] = value
    LETTER_CODES[key.lower()] = value

CHAR_CODES = {**DIGIT_CODES, **LETTER_CODES}
CHAR_CODES[" "] = ("", "")

T = TypeVar("T")


def _scale_grid(grid: Sequence[Sequence[T]], factor: int) -> list[list[T]]:
    """Repeat each row and column to scale the grid by ``factor``."""
    if factor <= 1:
        return [list(row) for row in grid]
    scaled: list[list[T]] = []
    for row in grid:
        expanded_row: list[T] = []
        for cell in row:
            expanded_row.extend([cell] * factor)
        for _ in range(factor):
            scaled.append(list(expanded_row))
    return scaled


def build_rectangular_mask(width: int, height: int) -> list[str]:
    """Return a simple rectangular mask with entry/exit gaps on opposite walls."""
    if width < MIN_MASK_WIDTH or height < MIN_MASK_HEIGHT:
        raise ValueError(
            f"Mask must be at least {MIN_MASK_WIDTH} columns by {MIN_MASK_HEIGHT} rows."
        )
    if width > MAX_MASK_WIDTH or height > MAX_MASK_HEIGHT:
        raise ValueError(
            f"Mask must be at most {MAX_MASK_WIDTH} columns by {MAX_MASK_HEIGHT} rows."
        )

    top_row = "#." + "#" * (width - 2)
    bottom_row = "#" * (width - 2) + ".#"
    middle_row = "#" + "." * (width - 2) + "#"

    rows = [top_row]
    for _ in range(height - 2):
        rows.append(middle_row)
    rows.append(bottom_row)
    return rows


def bitmasks_to_image(bm: Sequence[Sequence[int | None]], scale: int = 8) -> list[list[int]]:
    """Convert spanning-forest bitmasks to a binary wall/floor raster image."""
    H, W = len(bm), len(bm[0])
    grid = [[1] * (2 * W + 1) for _ in range(2 * H + 1)]

    for r in range(H):
        for c in range(W):
            x = bm[r][c]
            if x is None:
                continue
            grid[2 * r + 1][2 * c + 1] = 0
            if x & U:
                grid[2 * r][2 * c + 1] = 0
            if x & D:
                grid[2 * r + 2][2 * c + 1] = 0
            if x & L:
                grid[2 * r + 1][2 * c] = 0
            if x & R:
                grid[2 * r + 1][2 * c + 2] = 0

    return grid if scale == 1 else _scale_grid(grid, scale)


def maze_forest_from_mask(mask: Sequence[str], seed: int = 0, open_char: str = "."):
    """Generate a uniform spanning forest from a character mask."""
    H = len(mask)
    W = max(len(row) for row in mask)
    m = [row.ljust(W) for row in mask]

    # Build open-cell list with per-cell allowed-direction masks
    cells, allow = [], {}  # id -> (r,c), id -> allowed mask
    for r in range(H):
        for c in range(W):
            ch = m[r][c]
            if ch == open_char:
                vid = len(cells)
                cells.append((r, c))
                allow[vid] = 0xF
            else:
                try:
                    a = int(ch, 16)
                    vid = len(cells)
                    cells.append((r, c))
                    allow[vid] = a
                except Exception:
                    pass  # wall cell

    if not cells:
        raise ValueError("No open cells.")

    id_of = {rc: i for i, rc in enumerate(cells)}
    n = len(cells)

    # Build allowed undirected graph under mutual-allow constraint
    adj = [[] for _ in range(n)]

    def add_edge(a, b):
        adj[a].append(b)
        adj[b].append(a)

    for (r, c), a in id_of.items():
        for rr, cc, ba, bb in (
            (r - 1, c, U, D),
            (r + 1, c, D, U),
            (r, c - 1, L, R),
            (r, c + 1, R, L),
        ):
            b = id_of.get((rr, cc))
            if b is None or a >= b:
                continue
            if (allow[a] & ba) and (allow[b] & bb):
                add_edge(a, b)

    # Connected components (disconnection is allowed)
    comps = []
    seen = [False] * n
    for s in range(n):
        if seen[s]:
            continue
        q, comp = deque([s]), []
        seen[s] = True
        while q:
            v = q.popleft()
            comp.append(v)
            for nb in adj[v]:
                if not seen[nb]:
                    seen[nb] = True
                    q.append(nb)
        comps.append(comp)

    # Wilson's algorithm on each component -> uniform spanning forest
    rng = random.Random(seed)
    forest = set()

    for comp in comps:
        if len(comp) <= 1:
            continue
        comp_set = set(comp)

        # Local adjacency restricted to the component (kept simple)
        ladj = {v: [nb for nb in adj[v] if nb in comp_set] for v in comp}
        root = comp[-1]

        in_tree = {root}
        for start in comp:
            if start in in_tree:
                continue
            parent = {}
            v = start
            while v not in in_tree:
                if not ladj[v]:
                    # Isolated vertex cannot be connected within this component; treat as singleton.
                    break
                nv = rng.choice(ladj[v])
                parent[v] = nv  # overwriting erases loops implicitly
                v = nv
            v = start
            while v not in in_tree and v in parent:
                nv = parent[v]
                a, b = (v, nv) if v < nv else (nv, v)
                forest.add((a, b))
                in_tree.add(v)
                v = nv

    # Convert forest edges to per-cell bitmasks on the original canvas
    def has(a, b):
        if a > b:
            a, b = b, a
        return (a, b) in forest

    bm = [[None] * W for _ in range(H)]
    for (r, c), v in id_of.items():
        x = 0
        u = id_of.get((r - 1, c))
        x |= U if u is not None and has(v, u) else 0
        u = id_of.get((r + 1, c))
        x |= D if u is not None and has(v, u) else 0
        u = id_of.get((r, c - 1))
        x |= L if u is not None and has(v, u) else 0
        u = id_of.get((r, c + 1))
        x |= R if u is not None and has(v, u) else 0
        # Enforce "disallowed directions must be wall" (optional safety check)
        if (x & ~allow[v]) != 0:
            raise RuntimeError(
                "Internal error: generated an edge through a disallowed wall."
            )
        bm[r][c] = x

    return bm


def generate_mask_image(
    mask: Sequence[str] | None = None,
    seed: int = 0,
    scale: int = 1,
    open_char: str = ".",
 ) -> list[list[int]]:
    """Utility helper to reuse the demo maze generator elsewhere."""
    mask = mask or DEFAULT_MASK
    bm = maze_forest_from_mask(mask, seed=seed, open_char=open_char)
    return bitmasks_to_image(bm, scale=scale)


def _split_message_lines(message: str) -> list[str]:
    raw_lines = message.splitlines()
    if not raw_lines:
        raw_lines = [message]
    lines = []
    for line in raw_lines:
        normalized_chars = []
        for ch in line:
            if ch.isspace():
                normalized_chars.append(" ")
            else:
                normalized_chars.append(ch)
        lines.append("".join(normalized_chars))
    if not any(lines):
        return [""]
    return lines


def _lookup_tokens(ch: str) -> tuple[str, str]:
    key = ch.upper() if ch.isalpha() else ch
    tokens = CHAR_CODES.get(key)
    if tokens is None:
        raise ValueError(f"Unsupported character '{ch}'. Use digits, A-Z, or spaces.")
    return tokens


def _write_line_to_rows(
    rows: list[list[str]],
    row_top: int,
    row_bottom: int,
    line: str,
    glyph_spans: list[tuple[str, int, int, int, int]],
    start_col: int,
) -> None:
    cleaned = list(line)
    if not cleaned:
        return

    width = len(rows[0])
    col = start_col
    prev_width = 0

    for idx, ch in enumerate(cleaned):
        if idx > 0:
            col += prev_width + 1  # separator column

        top_code, bottom_code = _lookup_tokens(ch)
        glyph_width = max(len(top_code), len(bottom_code))

        if col < MESSAGE_MIN_COL:
            raise ValueError("Message is too long for the available columns.")

        if col + glyph_width - 1 >= width - 1:
            raise ValueError("Message exceeds the right boundary of the mask.")

        top_seq = top_code if top_code != "." else "." * glyph_width
        if len(top_seq) < glyph_width:
            top_seq = top_seq + "." * (glyph_width - len(top_seq))

        bottom_seq = bottom_code
        if len(bottom_seq) < glyph_width:
            bottom_seq = bottom_seq + "." * (glyph_width - len(bottom_seq))

        for offset in range(glyph_width):
            col_idx = col + offset
            rows[row_top][col_idx] = top_seq[offset]
            rows[row_bottom][col_idx] = bottom_seq[offset]

        prev_width = glyph_width
        glyph_spans.append((ch, row_top, row_bottom, col, glyph_width))


def _apply_message_to_mask(
    mask: Sequence[str],
    message: str,
    *,
    start_row: int = MESSAGE_START_ROW,
    start_col: int = MESSAGE_START_COL,
    line_gap: int = MESSAGE_LINE_GAP,
) -> tuple[list[str], list[tuple[int, int]], list[tuple[str, int, int, int, int]]]:
    rows = [list(row) for row in mask]
    lines = _split_message_lines(message)
    used_rows: list[tuple[int, int]] = []
    glyph_spans: list[tuple[str, int, int, int, int]] = []

    for line_idx, line in enumerate(lines):
        row_top = start_row + line_idx * (2 + line_gap)
        row_bottom = row_top + 1
        if row_bottom >= len(rows) - 1:
            raise ValueError("Message is too tall for the current mask.")
        _write_line_to_rows(rows, row_top, row_bottom, line, glyph_spans, start_col)
        used_rows.append((row_top, row_bottom))

    return ["".join(row) for row in rows], used_rows, glyph_spans


DEFAULT_WALL_COLOR = (12, 15, 18)
DEFAULT_MESSAGE_WALL_COLOR = (225, 36, 36)
DEFAULT_FLOOR_COLOR = (235, 232, 224)


@dataclass
class RenderOptions:
    message: str = "DUNGEON"
    message_start_row: int = MESSAGE_START_ROW
    message_start_col: int = MESSAGE_START_COL
    seed: int = 0
    mask_width: int = DEFAULT_MASK_WIDTH
    mask_height: int = DEFAULT_MASK_HEIGHT
    mask_override: list[str] | None = None
    wall_color: tuple[int, int, int] = DEFAULT_WALL_COLOR
    message_wall_color: tuple[int, int, int] = DEFAULT_MESSAGE_WALL_COLOR
    floor_color: tuple[int, int, int] = DEFAULT_FLOOR_COLOR
    scale: int = 12
    dpi: int = 500


def _build_base_image(
    grid: Sequence[Sequence[int]],
    wall_color: tuple[int, int, int],
    floor_color: tuple[int, int, int],
    *,
    highlight_mask: Sequence[Sequence[bool]] | None = None,
    highlight_color: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Expand the maze grid into an RGB ``Image``."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    image = Image.new("RGB", (width, height))
    pixels = image.load()

    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            pixels[x, y] = wall_color if value == 1 else floor_color

    if highlight_mask is not None and highlight_color is not None:
        for y, row in enumerate(highlight_mask):
            if y >= height:
                break
            for x, mark in enumerate(row):
                if x >= width:
                    break
                if mark:
                    pixels[x, y] = highlight_color

    return image


def _build_letter_wall_mask(
    mask: Sequence[str],
    used_rows: Sequence[tuple[int, int]],
    glyph_spans: Sequence[tuple[str, int, int, int, int]],
) -> list[list[bool]] | None:
    rows_to_scan = {idx for pair in used_rows for idx in pair}
    if not rows_to_scan:
        return None

    t_top_spans = [
        (row_top, col_start, width)
        for ch, row_top, _row_bottom, col_start, width in glyph_spans
        if ch in ("t", "T")
    ]

    height = len(mask)
    width = len(mask[0]) if mask else 0
    highlight_height = 2 * height + 1
    highlight_width = 2 * width + 1
    highlight = [[False] * highlight_width for _ in range(highlight_height)]

    for row_idx in rows_to_scan:
        if row_idx < 0 or row_idx >= height:
            continue
        row = mask[row_idx]
        for col_idx, ch in enumerate(row):
            try:
                allow_mask = int(ch, 16)
            except ValueError:
                continue
            blocked = (~allow_mask) & 0xF
            if blocked & U:
                target_row = 2 * row_idx
                for c in range(2 * col_idx, min(2 * col_idx + 3, highlight_width)):
                    highlight[target_row][c] = True
            if blocked & D:
                target_row = 2 * row_idx + 2
                for c in range(2 * col_idx, min(2 * col_idx + 3, highlight_width)):
                    highlight[target_row][c] = True
            if blocked & L:
                target_col = 2 * col_idx
                for r in range(2 * row_idx, min(2 * row_idx + 3, highlight_height)):
                    highlight[r][target_col] = True
            if blocked & R:
                target_col = 2 * col_idx + 2
                for r in range(2 * row_idx, min(2 * row_idx + 3, highlight_height)):
                    highlight[r][target_col] = True

    for row_top, col_start, span_width in t_top_spans:
        r_idx = 2 * row_top
        if r_idx >= highlight_height:
            continue
        for offset in range(span_width):
            c_idx = 2 * (col_start + offset)
            for delta in range(3):
                c = c_idx + delta
                if c < highlight_width:
                    highlight[r_idx][c] = False

    if not any(any(row) for row in highlight):
        return None
    return highlight


def render_message_maze(
    options: RenderOptions,
) -> tuple[
    Image.Image,
    list[list[int]],
    list[str],
    list[tuple[int, int]],
    list[tuple[str, int, int, int, int]],
]:
    """Generate a dungeon image whose topology hides the provided message."""
    base_mask = (
        options.mask_override
        if options.mask_override is not None
        else build_rectangular_mask(options.mask_width, options.mask_height)
    )
    mask_with_message, used_rows, glyph_spans = _apply_message_to_mask(
        base_mask,
        options.message,
        start_row=options.message_start_row,
        start_col=options.message_start_col,
    )
    raw = generate_mask_image(
        mask=mask_with_message, seed=options.seed, scale=options.scale
    )
    highlight_mask = _build_letter_wall_mask(mask_with_message, used_rows, glyph_spans)
    if highlight_mask is not None and options.scale > 1:
        highlight_mask = _scale_grid(highlight_mask, options.scale)
    base_img = _build_base_image(
        raw,
        options.wall_color,
        options.floor_color,
        highlight_mask=highlight_mask,
        highlight_color=options.message_wall_color,
    )
    return base_img, raw, mask_with_message, used_rows, glyph_spans


def render_to_base64(options: RenderOptions) -> str:
    """Return the hidden-message dungeon PNG encoded as a data URI fragment."""
    img, _, _, _, _ = render_message_maze(options)
    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=(options.dpi, options.dpi))
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _demo(seed: int = 0):
    opts = RenderOptions(seed=seed)
    img, _, _, _ = render_message_maze(opts)
    img.show()


__all__ = [
    "DEFAULT_MASK",
    "MESSAGE_START_ROW",
    "MESSAGE_LINE_GAP",
    "MESSAGE_START_COL",
    "MESSAGE_MAX_COLUMNS",
    "RenderOptions",
    "bitmasks_to_image",
    "generate_mask_image",
    "maze_forest_from_mask",
    "render_message_maze",
    "render_to_base64",
]


if __name__ == "__main__":
    _demo()
