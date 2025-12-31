## Dungeon Drawer (Flask)

Render Wilson mazes that hide a short message as a hex bitmask, then export a chunky PNG preview.

### Setup

```bash
uv sync
```

### Run (development)

```bash
uv run flask --app main run --debug
```

Visit http://127.0.0.1:5000/ and fill the form:

- Message accepts A-Z, 0-9, and spaces. Spaces insert a 2-column gap; leave the textarea blank to render a maze without a hidden message.
- Message row/column let you choose the 0-indexed mask coordinates for the text's top-left pixel.
- Mask width/height controls the grid size (min 5×6, max 80×80).
- Mask layout can be edited directly; the default mask rows are pre-filled so you can tweak the walls or paste your own (# for walls, . for open cells, 0-F for constrained openings). Leave it empty to keep the width/height controls in charge.
- Pixel scale multiplies the rendered PNG size (1-40); DPI metadata defaults to 500.
- Wall, glyph (message walls), and floor colors accept `#RRGGBB`.
- Preview shows the encoded mask rows and basic width/height/floor coverage stats.

Save the PNG directly from the preview image.
