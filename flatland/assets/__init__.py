"""Branding assets, and how to hand them to each thing that wants an icon.

Two consumers, two formats:

* the browser tab wants scalable vector art, and NiceGUI will inline an SVG
  *string* as a data URL - so no file needs serving;
* the native window goes through .NET's `Icon` on Windows, which will only read
  a real .ico. `flatland.ico` is generated from the sprite artwork by
  `tools/make_icon.py`.
"""

import atexit
from contextlib import ExitStack
from importlib.resources import as_file, files

_ICON_SVG = "flatland.svg"
_ICON_ICO = "flatland.ico"

# Installed from a zip, `as_file` materialises the icon in a temp file that is
# deleted when its context exits. Hold the context open for the life of the
# process, or the path we hand out goes stale under us.
_resources = ExitStack()
atexit.register(_resources.close)
_icon_path = None


def favicon_svg():
    """The icon as an SVG string, for NiceGUI's `favicon=`.

    NiceGUI only recognises a string as SVG when it starts with `<svg`, so the
    XML declaration and the exporter's comment are trimmed off the front.
    """
    svg = files(__package__).joinpath(_ICON_SVG).read_text(encoding="utf-8")
    start = svg.find("<svg")
    return svg[start:] if start != -1 else svg


def window_icon_path():
    """Filesystem path to the .ico, for pywebview's `icon=`.

    Returns None if it cannot be materialised on disk - an icon is decoration,
    and must never be the reason a window fails to open.
    """
    global _icon_path
    if _icon_path is None:
        try:
            resource = files(__package__).joinpath(_ICON_ICO)
            _icon_path = str(_resources.enter_context(as_file(resource)))
        except (FileNotFoundError, OSError):
            return None
    return _icon_path
