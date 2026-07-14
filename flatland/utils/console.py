"""Terminal notices that are hard to miss.

A plain `print()` scrolls past unnoticed in a wall of log output, so a message
like "open this URL to see the visualisation" simply does not land - people sit
staring at a terminal waiting for a window that is never going to appear. These
helpers draw the important ones as bordered panels instead.

Kept separate from the renderer so the whole project can speak with one voice.
"""

import sys
import threading

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# safe_box: fall back to ASCII borders on terminals that cannot draw box glyphs.
_console = Console(highlight=False, safe_box=True)

_quiet = False


def set_quiet(quiet=True):
    """Silence the panels and the live status display.

    One switch rather than a flag threaded through every call site: `--quiet`
    should mean "say nothing", and a caller adding a new notice should not have
    to remember to honour it.
    """
    global _quiet
    _quiet = quiet


def is_quiet():
    return _quiet


def _glyph(char, fallback=""):
    """`char` if the terminal can actually encode it, else `fallback`.

    A Windows console is often still cp1252, which has no emoji and no U+26A0.
    Printing one there raises UnicodeEncodeError - so a decoration meant to make
    a message impossible to miss would instead replace it with a traceback.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        char.encode(encoding)
    except UnicodeEncodeError, LookupError:
        return fallback
    return char


def _panel(body, title, border_style):
    if _quiet:
        return
    _console.print()
    _console.print(
        Panel(
            body,
            title=title,
            title_align="left",
            border_style=border_style,
            padding=(1, 2),
            expand=False,
        )
    )
    _console.print()
    _console.file.flush()


def info(message):
    """A one-line status. Prefixed, not boxed - these are allowed to scroll by."""
    if _quiet:
        return
    _console.print(rf"[dim]\[flatland][/dim] {message}")
    _console.file.flush()


def warn(message, title="WARNING"):
    title = _glyph("⚠ ", "! ") + title
    _panel(Text(message), title=f"[bold]{title}[/bold]", border_style="red")


class RenderStatus:
    """The renderer's panel, kept live so the connection state is visible.

    Without this, "nothing is happening" and "you never actually opened the
    link" look identical from the terminal. The dot answers the only question a
    student has at that moment: is it me, or is it broken?

    `get_viewers` is polled for the number of connected browsers/windows.
    """

    _DOT = _glyph("●", "*")

    def __init__(self, url, port, get_viewers, native=False, reason=None, remote=False):
        self.url = url
        self.port = port
        self.get_viewers = get_viewers
        self.native = native
        self.reason = reason
        self.remote = remote

        self._seen_viewer = False
        self._live = None
        self._stop = threading.Event()
        self._thread = None

    # -- the status line ----------------------------------------------------
    def _status(self):
        try:
            n = self.get_viewers()
        except Exception:  # noqa: BLE001 - status must never break the sim
            n = 0

        if n:
            self._seen_viewer = True
            return "green", (
                "Connected"
                if self.native
                else f"Connected - {n} viewer{'s' if n > 1 else ''} watching"
            )
        if self._seen_viewer:
            return "red", "Disconnected"
        if self.native:
            return "yellow", "Opening the window ..."
        return "yellow", "Waiting for you to open the link ..."

    def _render(self):
        color, message = self._status()

        body = Text()
        if self.native:
            body.append("Showing the visualisation in a desktop window.\n")
            body.append("Closing that window will end this program.\n\n", style="dim")
            body.append("Also viewable at ", style="dim")
            body.append(f"{self.url}\n", style="cyan underline")
        else:
            body.append(
                "The visualisation is running, but no window could be opened.\n\n"
            )
            body.append("Open this link in your browser to watch it:\n\n", style="bold")
            body.append(f"    {self.url}\n", style="bold cyan underline")
            if self.remote:
                body.append(
                    f"\nOn a different machine? Use this machine's address, port "
                    f"{self.port}.\n",
                    style="dim",
                )
            if self.reason:
                body.append(
                    f"\nWhy there is no window: {self.reason}\n", style="dim italic"
                )

        body.append("\n")
        body.append(f"{self._DOT} ", style=f"bold {color}")
        body.append(message, style=color)

        if self.native:
            title = _glyph("\U0001f682 ", "") + "FLATLAND"
            border = "green"
        else:
            title = _glyph("\U0001f440 ", "") + "OPEN THIS LINK IN YOUR BROWSER"
            border = "yellow"
        return Panel(
            body,
            title=f"[bold]{title}[/bold]",
            title_align="left",
            border_style=border,
            padding=(1, 2),
            expand=False,
        )

    # -- lifecycle ----------------------------------------------------------
    def start(self):
        if _quiet:
            return
        # A redrawing panel needs a real terminal. Piped to a file or a CI log it
        # would emit one frame per refresh, so print it once and leave it.
        if not _console.is_terminal:
            _console.print()
            _console.print(self._render())
            _console.print()
            _console.file.flush()
            return

        _console.print()
        self._live = Live(
            self._render(), console=_console, refresh_per_second=8, transient=False
        )
        self._live.start()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.wait(0.25):
            # stop() may have torn the Live down between the wait and here.
            live = self._live
            if live is None:
                return
            live.update(self._render())

    def stop(self):
        """Must be called before the process exits, or the terminal is left with
        a hidden cursor and a half-drawn panel."""
        self._stop.set()
        if self._live is not None:
            self._live.update(self._render())
            self._live.stop()
            self._live = None
