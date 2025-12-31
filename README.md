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

- Message accepts A-Z, 0-9, and spaces. Spaces insert a 2-column gap.
- Mask width/height controls the grid size (min 5×6, max 80×80).
- Pixel scale multiplies the rendered PNG size (1-40); DPI metadata defaults to 500.
- Wall/Floor colors accept `#RRGGBB`.
- Preview shows the encoded mask rows and basic width/height/floor coverage stats.

Save the PNG directly from the preview image.
