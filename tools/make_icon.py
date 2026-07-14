"""Regenerate flatland/assets/flatland.ico from the sprite artwork.

The window icon has to be a real .ico: on Windows pywebview loads it through
.NET's `Icon`, which reads nothing else. The browser tab does not need this - it
gets the SVG inlined as a data URL - so this only exists for the native window.

Rasterised from the PNG rather than the SVG so that no SVG renderer is needed to
build the project; the PNG is the same artwork, already exported at 300x300.

    uv run python tools/make_icon.py
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "flatland" / "png" / "Scenery-Bergwelt_A_Teil_2_mitte.png"
TARGET = ROOT / "flatland" / "assets" / "flatland.ico"

# Ship every size Windows might ask for (taskbar, alt-tab, title bar, explorer)
# rather than making it downscale one for us.
SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

if __name__ == "__main__":
    Image.open(SOURCE).convert("RGBA").save(TARGET, format="ICO", sizes=SIZES)
    print(f"wrote {TARGET.relative_to(ROOT)} with sizes {[s[0] for s in SIZES]}")
