"""Browser-based graphics layer, built on NiceGUI.

Replaces the old pyglet window. PILSVG still does all the drawing; this class
only owns the "window", which here is a page served over HTTP. That means a
headless machine (training box, CI runner, container) can be watched from
another machine's browser, which a local GL window could never do.

The simulation loop stays synchronous. `show()` drops the freshly composited
frame into a slot; the server, running on a daemon thread, polls that slot and
pushes it to any connected browsers. If nobody is watching, `show()` skips the
PNG encode entirely and costs almost nothing.
"""

import atexit
import base64
import importlib.util
import io
import os
import socket
import subprocess
import sys
import threading
import time
from queue import Empty, Queue

from flatland import assets
from flatland.utils import console
from flatland.utils.graphics_pil import PILSVG

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080

# How many ports to try, upward from DEFAULT_PORT, when none was asked for.
# Large enough that you would have to be hosting a small farm to exhaust it,
# small enough that exhausting it means something is wrong and we should say so
# rather than scan into the ephemeral range.
PORT_SCAN = 64

# How long to wait before deciding the native window came up. It dies almost
# immediately when it is going to fail (no display, no webview runtime), so this
# only needs to outlast process startup.
NATIVE_STARTUP_GRACE = 4.0

# No frame asked for in this long, and we call the run paused. Generous, so a
# merely slow step (a big env, a thinking policy) is not mislabelled.
PAUSED_AFTER = 1.5

# Every viewer gone for this long, and the run ends. Comfortably longer than a
# page reload or a dropped-and-resumed socket, both of which briefly show zero
# viewers: mistaking either for "nobody is watching" would kill the run under
# someone who is very much still there.
VIEWER_GRACE = 1.0

# One server per process, shared by every RenderTool. Registering a route per
# renderer would mean mutating the route table after uvicorn has started; the
# pages below are static routes that read this registry at connection time.
_renderers = {}
_server = None
_server_lock = threading.Lock()


def _local_url(host, port):
    """A URL that actually resolves from this machine.

    0.0.0.0 means "bind every interface"; it is not a routable destination, so
    it must not be handed to a browser or a webview.
    """
    return f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"


ALIVE_ROUTE = "/_flatland/alive"

# The run is over -> get rid of the view.
#
# Killing the process we launched is not enough to shut a window: for a Chromium
# app window that process is only a launcher (and under WSL it is an interop
# shim), so the window outlives it. Letting the page notice the server has gone
# and close itself is simpler and stricter - it also works when the simulation is
# killed outright and never gets a chance to clean anything up.
#
# window.close() only works on windows that script opened, which is exactly the
# ones we opened. Browsers refuse it for a tab the user opened themselves (I
# checked), so for those the page is replaced with an unmistakable end state
# instead of leaving a frozen last frame that looks like a hung simulation.
# NiceGUI overlays a "Connection lost / Trying to reconnect..." box whenever the
# socket blips. For a viewer of a simulation that is just noise: a brief drop
# self-heals, and a permanent one is already handled - the page either closes
# itself or says the run ended, which is the useful message.
_HIDE_RECONNECT_CSS = """
<style>
  .nicegui-error-popup { display: none !important; }
</style>
"""

_SELF_CLOSE_JS = f"""
<script>
(() => {{
  let misses = 0;
  const ended = () => {{
    window.close();  // silently ignored for a tab the user opened
    setTimeout(() => {{
      document.title = 'Flatland - ended';
      document.body.innerHTML =
        '<div style="position:fixed;inset:0;display:flex;align-items:center;' +
        'justify-content:center;flex-direction:column;gap:.5rem;' +
        'font-family:ui-monospace,monospace;color:#6b7280;background:#f8f8f8">' +
        '<div style="font-size:1.1rem;font-weight:600">Simulation ended</div>' +
        '<div style="font-size:.85rem">You can close this tab.</div></div>';
    }}, 400);
  }};
  const timer = setInterval(async () => {{
    try {{
      const r = await fetch('{ALIVE_ROUTE}', {{cache: 'no-store'}});
      misses = r.ok ? 0 : misses + 1;
    }} catch (e) {{
      misses += 1;
    }}
    // Several in a row, not one: a single miss is a hiccup, not a dead server.
    if (misses >= 3) {{ clearInterval(timer); ended(); }}
  }}, 1000);
}})();
</script>
"""


def _is_wsl():
    """Are we a Linux process inside WSL?

    WSL has no Linux GUI to speak of for our purposes, but it *can* run Windows
    executables - so the best window available is a Windows one.
    """
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/sys/kernel/osrelease") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


# Chromium in --app mode is the closest thing to a webview binary that is simply
# *there* on a Windows box: Edge ships with the OS.
_WINDOWS_BROWSER_APPS = ("msedge.exe", "chrome.exe")

# Only reached if the registry lookup below fails.
_WINDOWS_BROWSER_FALLBACKS = (
    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe",
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
)


def _run_windows(args, timeout=15):
    """Run a Windows command from inside WSL and return its stdout."""
    try:
        # cwd=/mnt/c: run from a path Windows can represent, or cmd.exe warns and
        # refuses to expand anything.
        out = subprocess.run(
            args, cwd="/mnt/c", capture_output=True, text=True, timeout=timeout
        )
        return out.stdout.replace("\r", "")
    except OSError, subprocess.SubprocessError:
        return ""


def _windows_env(name, default):
    """Read a Windows environment variable from inside WSL."""
    value = _run_windows(["cmd.exe", "/c", f"echo %{name}%"]).strip()
    return value if value and not value.startswith("%") else default


def _find_windows_browser():
    """Locate a Windows Chromium, the way Windows itself would.

    `start msedge` resolves through the App Paths registry key rather than PATH,
    so we ask the same question - no guessing at install directories. We still
    launch the exe ourselves rather than going through `start`: `start` detaches,
    and we need a real process handle to know the window opened and to notice
    when it closes.
    """
    for app in _WINDOWS_BROWSER_APPS:
        for hive in ("HKLM", "HKCU"):
            key = rf"{hive}\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{app}"
            out = _run_windows(["reg.exe", "query", key, "/ve"])
            for line in out.splitlines():
                if "REG_SZ" not in line:
                    continue
                win_path = line.split("REG_SZ", 1)[1].strip()
                path = _run_windows(["wslpath", "-u", win_path]).strip()
                if path and os.path.isfile(path):
                    return path

    return next((p for p in _WINDOWS_BROWSER_FALLBACKS if os.path.isfile(p)), None)


def _try_open_wsl_window(url, title="Flatland", size=(400, 400)):
    """Open a *Windows* window on `url`, from inside WSL.

    Returns (process, None) on success, or (None, reason) on failure.

    pywebview cannot help here - there is no Linux GUI stack to draw on - but WSL
    can launch Windows executables, and Edge in `--app` mode is a chromeless
    window that ships with Windows. It picks up the page's favicon as its window
    icon, so it gets our branding for free.

    The dedicated --user-data-dir is not optional. Without it Edge hands the URL
    to the user's already-running Edge and our launcher exits immediately: the
    window would open inside their normal browser session, and we would lose the
    process handle that `exit_on_close` depends on.
    """
    exe = _find_windows_browser()
    if exe is None:
        return None, "no Windows Edge or Chrome found to open a window with"

    profile = _windows_env("TEMP", "C:\\Windows\\Temp") + "\\flatland-window"

    try:
        proc = subprocess.Popen(
            [
                exe,
                f"--app={url}",
                f"--user-data-dir={profile}",
                f"--window-size={size[0]},{size[1]}",
                "--no-first-run",
                "--no-default-browser-check",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as e:
        return None, str(e)

    deadline = time.monotonic() + NATIVE_STARTUP_GRACE
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return (
                None,
                f"the Windows window exited immediately (code {proc.returncode})",
            )
        time.sleep(0.1)
    return proc, None


def _try_open_native_window(url, title="Flatland", size=(400, 400)):
    """Try to open a pywebview window on `url`, in its own process.

    Returns (process, None) on success, or (None, reason) on failure.

    Deliberately NOT nicegui's own native mode (ui.run(native=True)):
      - with pywebview missing it calls sys.exit(1), which would take the render
        server down with it;
      - when the window is closed it calls core.stop_and_exit() -> os._exit,
        which would kill the training run. Closing a viewer must never do that.

    A fresh interpreter, rather than multiprocessing: 'spawn' re-imports the
    parent's __main__, so a training script without an `if __name__ ==
    "__main__"` guard would run itself a second time. This child imports nothing
    of the caller's.
    """
    if importlib.util.find_spec("webview") is None:
        return None, "pywebview is not installed (pip install pywebview)"

    # pywebview takes the icon on start(), not create_window(). On Windows it is
    # loaded via .NET's Icon, which reads .ico and nothing else.
    icon = assets.window_icon_path()
    start_args = f"icon={icon!r}" if icon else ""

    # The page's own window.close() does not reliably shut a pywebview window, so
    # the child watches the server itself and destroys the window when it stops
    # answering - the same contract as the browser window, enforced differently.
    alive_url = url.rsplit("/", 1)[0] + ALIVE_ROUTE

    code = (
        "import threading, time, urllib.request, webview\n"
        f"w = webview.create_window({title!r}, {url!r}, "
        f"width={size[0]}, height={size[1]})\n"
        "def _watch():\n"
        "    misses = 0\n"
        "    while True:\n"
        "        time.sleep(1)\n"
        "        try:\n"
        f"            urllib.request.urlopen({alive_url!r}, timeout=2).read()\n"
        "            misses = 0\n"
        "        except Exception:\n"
        "            misses += 1\n"
        "        if misses >= 3:\n"
        "            try:\n"
        "                w.destroy()\n"
        "            except Exception:\n"
        "                pass\n"
        "            return\n"
        "threading.Thread(target=_watch, daemon=True).start()\n"
        f"webview.start({start_args})\n"
    )
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as e:
        return None, str(e)

    # If it is going to fail (headless box, no GTK/Qt/WebView2), it fails fast.
    deadline = time.monotonic() + NATIVE_STARTUP_GRACE
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            # stderr is a pipe (see Popen above), but Popen.stderr is Optional.
            raw = proc.stderr.read() if proc.stderr is not None else b""
            err = (raw or b"").decode(errors="replace").strip()
            reason = (
                err.splitlines()[-1] if err else f"exited with code {proc.returncode}"
            )
            return None, reason
        time.sleep(0.1)
    return proc, None


def _bind_error(host, port):
    """Is `port` ours to take? Returns None if it is, or the OSError if not."""
    probe = socket.socket()
    try:
        # No SO_REUSEADDR: we want this to fail exactly when uvicorn's bind would.
        probe.bind((host, port))
    except OSError as e:
        return e
    finally:
        probe.close()
    return None


def _claim_port(host, port):
    """Settle on a port to serve on, before uvicorn gets a say. Returns it.

    We probe rather than letting uvicorn discover the problem, because uvicorn
    runs the app's lifespan startup hooks *before* it binds the socket, so our
    `started` event fires even when the bind then fails. Waiting on that event
    therefore cannot tell a healthy server from one that never bound - and the
    failure mode is nasty: we would print a URL that some *other* process is
    answering on, and the user would watch a stale render and never know.

    `port=None` means "any port", and we scan upward from DEFAULT_PORT. A box
    where 8080 is already taken is entirely ordinary - another dev server, or a
    second flatland run in the next terminal - and there is nothing there for
    the user to decide, so deciding for them beats failing at them.

    An explicit port, on the other hand, is a request and not a hint: the caller
    may well have picked it because something *else* is expecting the render on
    exactly that port. Quietly serving on a different one would break that
    silently, so a taken port stays an error.

    Note the gap between probing and uvicorn's own bind: nothing stops another
    process taking the port in between. Nothing can - the loser of that race has
    to lose somewhere - and it is a good deal narrower than the alternative.
    """
    if port is not None:
        err = _bind_error(host, port)
        if err is None:
            return port
        raise RuntimeError(
            f"The flatland render server cannot use port {port} - something else "
            f"is already listening there ({err.strerror or err}). "
            f"Choose another one, e.g. --port {port + 1}, or leave the port unset "
            f"to have a free one picked for you."
        ) from None

    for candidate in range(DEFAULT_PORT, DEFAULT_PORT + PORT_SCAN):
        if _bind_error(host, candidate) is None:
            return candidate

    raise RuntimeError(
        f"The flatland render server could not find a free port in "
        f"{DEFAULT_PORT}-{DEFAULT_PORT + PORT_SCAN - 1}: every one of them is "
        f"already in use. Free one up, or name one yourself with --port."
    )


class _Server:
    def __init__(self, host, port):
        # Imported here, on the calling thread, so a missing dependency surfaces
        # as a plain ImportError instead of a traceback buried in the server
        # thread followed by an opaque "failed to start".
        import nicegui  # noqa: F401

        self.host = host
        # None means "any"; the port we actually got is the one to report.
        self.port = _claim_port(host, port)
        self.started = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # Don't let the sim race ahead of uvicorn's bind.
        in_time = self.started.wait(timeout=30)
        if self._error or not in_time:
            raise RuntimeError(
                f"The flatland render server could not start on {self.host}:{self.port} "
                f"({self._error or 'timed out'})."
            )

    @property
    def clients(self):
        """Browsers currently holding a live socket.

        Read from NiceGUI's own client registry rather than counted via
        on_connect/on_disconnect: a client that goes away without a clean
        disconnect would otherwise leave the count permanently too high, and we
        gate frame encoding on it.
        """
        from nicegui import Client

        return sum(1 for c in Client.instances.values() if c.has_socket_connection)

    def wait_for_client(self, timeout=None):
        deadline = time.time() + (timeout if timeout is not None else 1e9)
        while time.time() < deadline:
            if self.clients:
                return True
            time.sleep(0.1)
        return False

    def _run(self):
        try:
            self._serve()
        except BaseException as e:  # noqa: BLE001 - reported to the calling thread
            self._error = f"{type(e).__name__}: {e}"
            self.started.set()

    def _serve(self):
        from nicegui import app, ui

        def fit_page():
            """Let the frame use the whole viewport.

            NiceGUI wraps page content in `.nicegui-content`, which ships with
            padding and a flex gap. Against a frame already sized to the full
            viewport height that padding is what pushes the page over the edge
            and raises scrollbars, so strip it.
            """
            ui.query("body").style(
                "margin:0; padding:0; overflow:hidden;" " background:#f8f8f8;"
            )
            ui.query(".nicegui-content").classes("p-0 m-0 gap-0").style(
                "height:100dvh; width:100vw; align-items:center; justify-content:center;"
            )

        # Height of the metric strip. The frame is capped at the viewport MINUS
        # this, so adding the strip cannot bring the scrollbars back.
        BAR = "2rem"

        def metrics(renderer, fit):
            """A strip of counters above the frame."""
            with ui.row().style(
                f"height:{BAR}; gap:1.25rem; align-items:center; padding:0 .75rem;"
                " font-family:ui-monospace,monospace; font-size:.78rem;"
                " color:#4b5563; width:100%; justify-content:center;"
                " flex:0 0 auto;"
                if fit
                else "gap:1.25rem;"
            ):
                state = ui.label()
                step = ui.label()
                arrived = ui.label()
                fps = ui.label()

            def poll():
                s = renderer.stats()
                playing = s.get("playing")
                state.set_text(("▶ playing" if playing else "⏸ paused"))
                state.style(
                    f"font-weight:600; color:{'#15803d' if playing else '#b45309'}"
                )

                n, cap_ = s.get("step"), s.get("max_steps")
                step.set_text(
                    f"step {n}{f' / {cap_}' if cap_ else ''}" if n is not None else ""
                )

                total = s.get("agents") or 0
                arrived.set_text(
                    f"arrived {s.get('done', 0)} / {total}" if total else ""
                )

                fps.set_text(f"{s['fps']:.0f} fps" if playing and s.get("fps") else "")

            ui.timer(0.25, poll)

        def view(renderer, label=None, fit=True):
            if label:
                ui.label(label).classes("text-sm font-mono opacity-60")

            metrics(renderer, fit)

            img = ui.interactive_image()
            # Fit inside the viewport, preserve aspect, and never scale beyond
            # natural size: upscaling a raster frame is what makes the art look
            # soft. Render at a higher cell_size / screen size for more detail.
            # 100dvh, not 100vh: on mobile the browser chrome makes vh taller
            # than what is actually visible, which would reintroduce the scroll.
            cap = (
                f"max-height:calc(100dvh - {BAR}); max-width:100vw; min-height:0;"
                if fit
                else "max-width:100%;"
            )
            img.style(
                cap + " width:auto; height:auto; object-fit:contain; margin:0 auto;"
            )
            img.props("fit=scale-down")
            seen = {"v": -1}

            def poll():
                v, uri = renderer.latest_frame()
                if uri is not None and v != seen["v"]:
                    seen["v"] = v
                    img.set_source(uri)

            ui.timer(1 / 30, poll)

            def on_key(e):
                if e.action.keydown:
                    renderer.push_key(str(e.key))

            ui.keyboard(on_key=on_key)

        def show_all():
            # Every view, not just the window we opened: when the run ends, the
            # last frame otherwise sits there looking like a hung simulation.
            ui.add_head_html(_HIDE_RECONNECT_CSS + _SELF_CLOSE_JS)
            if not _renderers:
                ui.label("No flatland environment is rendering yet.")
                return
            if len(_renderers) == 1:
                fit_page()
                view(next(iter(_renderers.values())))
                return
            # Several envs stacked: they cannot all fit, so let the page scroll
            # normally rather than squeezing each one into a sliver.
            for key in sorted(_renderers):
                view(_renderers[key], label=f"env {key}", fit=False)

        @app.get(ALIVE_ROUTE)
        def alive():
            """Heartbeat for the window page. Its absence is the shutdown signal."""
            return {"alive": True}

        @ui.page("/")
        def index():
            show_all()

        @ui.page("/window")
        def window():
            """What the window we opened points at.

            Identical to `/` today - it stays a separate route so the window can
            be told apart from a hand-opened tab if that ever needs to differ.
            """
            show_all()

        @ui.page("/env/{idx}")
        def single(idx: int):
            ui.add_head_html(_HIDE_RECONNECT_CSS + _SELF_CLOSE_JS)
            renderer = _renderers.get(idx)
            if renderer is None:
                ui.label(f"No environment {idx}.")
                return
            fit_page()
            view(renderer)

        app.on_startup(self.started.set)

        ui.run(
            host=self.host,
            port=self.port,
            show=False,
            reload=False,
            title="Flatland",
            # NiceGUI inlines an SVG string as a data URL, so the tab icon stays
            # sharp at every size and nothing extra has to be served.
            favicon=assets.favicon_svg(),
            reconnect_timeout=10,
        )


def _ensure_server(host, port):
    global _server
    with _server_lock:
        if _server is None:
            _server = _Server(host, port)
        elif host != _server.host or (port is not None and port != _server.port):
            # uvicorn is already bound; a second host/port in one process would
            # need a second server. Say so rather than silently ignoring it.
            #
            # An unset port asks for "somewhere", and the running server is a
            # perfectly good somewhere - so it is not a conflict and gets no
            # warning. Only an explicit request we cannot honour is worth one.
            console.warn(
                f"A render server is already running on {_server.host}:{_server.port}.\n"
                f"The request for {host}:{port} is being ignored - this environment\n"
                f"will appear on the existing server instead.",
                title="PORT ALREADY IN USE",
            )
    return _server


class WEBGL(PILSVG):
    """Serve the rendered environment to a browser instead of a local window."""

    def __init__(
        self,
        width,
        height,
        jupyter=False,
        screen_width=800,
        screen_height=600,
        cell_size=None,
        host=None,
        port=None,
        wait_for_client=None,
        image_format="jpeg",
        quality=90,
        max_fps=30,
        native=True,
        exit_on_close=True,
    ):
        """
        native
            Try to open a native desktop window on the rendered page (requires
            pywebview). If that fails - headless box, no display, pywebview not
            installed - the render server keeps running regardless, and the URL
            to open in a browser is printed instead. Set False to never attempt
            it.
        wait_for_client
            Hold the first frame until someone is actually watching, so you see
            the run from step 0 rather than opening the link to find the episode
            half over. Defaults to whatever `exit_on_close` is: the run then
            lasts exactly as long as a viewer does, from both ends. Times out
            after 300s and carries on regardless.
        exit_on_close
            The run ends when you stop watching it (default True). That means
            closing the native window, and also the *last* viewer disconnecting -
            a browser tab counts. It only ever applies once someone has actually
            connected: a run nobody has looked at yet is left alone, so this
            cannot kill a job on a headless box that is simply waiting for you.

            Several viewers are fine: only the last one leaving ends the run, and
            a page reload or a dropped socket is tolerated for a few seconds
            first, so neither is mistaken for you walking away.

            Set False when the render is a side-view onto a job that must
            survive - a long training run you want to peek at and leave.

            Note this hard-exits: `finally` blocks in the caller do not run,
            though atexit handlers do. Use False if you have cleanup to do.
        image_format
            "jpeg" (default) or "png". This is the *transport* format for the
            live browser stream only; get_image() and save_image() are
            unaffected and remain lossless. JPEG is ~7x faster to encode and the
            artefacts are invisible at normal viewing; use "png" if you need the
            browser to show exactly what the compositor produced.
        quality
            JPEG quality, 1-95. Ignored for PNG.
        max_fps
            Ceiling on how often a frame is encoded and pushed. A sim stepping
            faster than this simply drops frames rather than burning CPU
            encoding ones no browser will ever display. None = no limit.
        """
        super().__init__(width, height, jupyter, screen_width, screen_height, cell_size)

        self.host = host or os.environ.get("FLATLAND_RENDER_HOST", DEFAULT_HOST)
        # Stays None when nobody named a port: that is not "8080", it is "any",
        # and only _claim_port - once it has probed - can turn it into a number.
        # open_window writes the port we actually got back over this.
        requested_port = port if port is not None else os.environ.get("FLATLAND_RENDER_PORT")
        self.port = int(requested_port) if requested_port is not None else None
        self.native = native and os.environ.get("FLATLAND_RENDER_NATIVE", "1") != "0"
        self.exit_on_close = exit_on_close

        # Follows exit_on_close by default, because the two are the same promise
        # seen from either end: the run lasts exactly as long as someone is
        # watching it. Starting without waiting would mean opening the link to
        # find the episode already half over - and then the run would still end
        # when you left. A side-view onto a long job (exit_on_close=False) must
        # never block, so it does not wait either.
        self.wait_for_client_on_open = (
            exit_on_close if wait_for_client is None else wait_for_client
        )
        self._native_proc = None

        self.image_format = image_format.lower()
        self.quality = quality
        self.max_fps = max_fps

        self.window_open = False
        self.closed = False
        self._idx = len(_renderers)
        self._frame_lock = threading.Lock()
        self._png = None
        self._mime = "image/jpeg" if self.image_format == "jpeg" else "image/png"
        self._version = 0
        self._last_encode = 0.0
        self._keys = Queue()
        self._status = None
        self._stats = {}
        self._last_show = 0.0
        self._fps = 0.0

    # -- window lifecycle ----------------------------------------------------
    def open_window(self):
        assert self.window_open is False, "Window is already open!"
        server = _ensure_server(self.host, self.port)
        # We may have asked for "any" (or been redirected onto an already-running
        # server); either way the port below is the real, bound one, and callers
        # reading self.port - get_endpoint_URL among them - need that, not None.
        self.port = server.port
        _renderers[self._idx] = self
        self.window_open = True

        url = _local_url(server.host, server.port)

        opened_native = False
        # Only set when opening a window was *attempted and failed*. Staying None
        # when native=False keeps the panel from reporting a deliberate choice as
        # though it were an error.
        reason = None
        if self.native:
            size = (
                (self.widthPx // 2) + 16,
                (self.heightPx // 2) + 39,
            )  # + window chrome

            # The window gets the self-closing page; the URL we print for humans
            # stays the plain one.
            window_url = url + "window"

            if _is_wsl():
                # WSL has no Linux GUI stack worth targeting, but it can launch
                # Windows executables - so the window comes from the Windows side.
                # Windows reaches the server over WSL's loopback forwarding, which
                # only covers sockets bound to the wildcard address.
                if server.host not in ("0.0.0.0", "::"):
                    reason = (
                        f"in WSL, a Windows window cannot reach a server bound to "
                        f"{server.host}. Bind 0.0.0.0 (the default) instead."
                    )
                else:
                    self._native_proc, reason = _try_open_wsl_window(
                        window_url, size=size
                    )
            else:
                self._native_proc, reason = _try_open_native_window(
                    window_url, size=size
                )

            opened_native = self._native_proc is not None
            if opened_native and self.exit_on_close:
                # Redundant with the viewer watchdog below - closing the window
                # also drops its socket - but it fires at once instead of after
                # the grace period, which is what you want when you deliberately
                # shut the window in front of you.
                threading.Thread(
                    target=self._exit_when_window_closed,
                    args=(self._native_proc,),
                    daemon=True,
                ).start()

        if self.exit_on_close:
            threading.Thread(
                target=self._exit_when_viewers_leave,
                args=(server,),
                daemon=True,
            ).start()

        # The panel stays live for the run: without a connection indicator,
        # "nothing is happening yet" and "you never opened the link" look
        # identical from the terminal, which is exactly where people get stuck.
        self._status = console.RenderStatus(
            url,
            server.port,
            get_viewers=lambda: server.clients,
            get_stats=self.stats,
            native=opened_native,
            reason=None if opened_native else reason,
            remote=server.host in ("0.0.0.0", "::"),
        )
        self._status.start()
        # A script that simply runs off the end never calls close_window(), and
        # a live display left running restores neither the cursor nor the line.
        atexit.register(self._stop_status)

        if self.wait_for_client_on_open and server.clients == 0:
            if not server.wait_for_client(timeout=300):
                console.warn(
                    "No viewer connected after 300 seconds. Continuing without one - "
                    "nothing will be displayed.",
                    title="NOBODY IS WATCHING",
                )

    def _exit_when_window_closed(self, proc):
        """End the process when the user closes the native window."""
        proc.wait()
        if self.closed:
            return  # close_window() killed it on purpose; not a user close.
        self._end_run("Window closed - exiting.")

    def _exit_when_viewers_leave(self, server):
        """End the run when the last viewer goes away.

        Guarded three ways, because getting this wrong kills someone's job:
          - it never fires until a viewer has actually connected, so a run
            waiting patiently on a headless box is never mistaken for abandoned;
          - it needs *every* viewer gone, not just one of several;
          - and it tolerates a few seconds of zero viewers first, because a page
            reload and a briefly-dropped socket both look exactly like that.
        """
        seen_viewer = False
        empty_since = None

        while not self.closed:
            time.sleep(0.5)
            if server.clients:
                seen_viewer = True
                empty_since = None
                continue
            if not seen_viewer:
                continue  # nobody has ever watched; nothing to walk away from
            now = time.monotonic()
            if empty_since is None:
                empty_since = now
            elif now - empty_since >= VIEWER_GRACE:
                self._end_run("Viewer disconnected - exiting.")
                return

    def _end_run(self, message):
        """Hard-exit the process. The main thread is parked in the caller's
        step/render loop, so there is nothing to politely unwind to - the same
        thing NiceGUI itself does. atexit handlers still run.
        """
        # Tear the live panel down first: os._exit would otherwise leave the
        # terminal with a hidden cursor and a half-drawn box.
        self._stop_status()
        console.info(message)
        sys.stdout.flush()
        sys.stderr.flush()
        atexit._run_exitfuncs()
        os._exit(0)

    def _stop_status(self):
        if self._status is not None:
            self._status.stop()
            self._status = None

    def close_window(self):
        # Set before terminating, so the watcher above can tell a deliberate
        # close_window() apart from the user closing the window.
        self.closed = True
        self._stop_status()
        if self._native_proc is not None and self._native_proc.poll() is None:
            self._native_proc.terminate()
        self._native_proc = None
        _renderers.pop(self._idx, None)
        self.window_open = False

    # -- run stats -----------------------------------------------------------
    def set_stats(self, stats):
        """Called by the renderer each frame with where the run has got to."""
        with self._frame_lock:
            self._stats = stats

    def stats(self):
        """Run stats plus what only this layer knows: is it moving, and how fast.

        Read from the server thread and from the terminal panel, so everything it
        touches is taken under the lock.
        """
        with self._frame_lock:
            stats = dict(self._stats)
            last = self._last_show
            fps = self._fps

        # "Paused" is inferred, not declared: there is no pause API - the caller
        # simply stops calling render_env. From here that is indistinguishable
        # from a very slow step, so allow a grace period before saying so.
        idle = time.monotonic() - last if last else None
        stats["playing"] = idle is not None and idle < PAUSED_AFTER
        stats["fps"] = round(fps, 1) if fps else 0.0
        return stats

    # -- frame plumbing ------------------------------------------------------
    def latest_frame(self):
        """(version, data-uri) - called from the server thread."""
        with self._frame_lock:
            return self._version, self._png

    def has_viewers(self):
        return _server is not None and _server.clients > 0

    def _tick(self):
        """Record that the sim asked for a frame.

        Deliberately before the no-viewer and max_fps early-returns below: this
        measures how fast the *simulation* is running, which is what "playing"
        and the step counter are about. Counting only encoded frames would report
        0 fps and "paused" for a sim that is running perfectly well with nobody
        watching, or one capped by max_fps.
        """
        now = time.monotonic()
        with self._frame_lock:
            if self._last_show:
                delta = now - self._last_show
                if delta > 0:
                    # Smoothed, or it jitters unreadably at speed.
                    instant = 1.0 / delta
                    self._fps = (
                        instant if not self._fps else (0.8 * self._fps + 0.2 * instant)
                    )
            self._last_show = now

    def show(self, block=False, from_event=False):
        if not self.window_open:
            self.open_window()
        if self.closed:
            return
        self._tick()
        # Nobody is watching: compositing and encoding the frame would be pure
        # waste, and a training run with an idle renderer attached is common.
        if not self.has_viewers():
            return
        # A sim stepping faster than the browser can paint gains nothing from
        # encoding every frame. Drop the ones nobody would see.
        now = time.monotonic()
        if self.max_fps and (now - self._last_encode) < (1.0 / self.max_fps):
            return
        self._last_encode = now

        pil_img = self.alpha_composite_layers().convert("RGB")
        buf = io.BytesIO()
        if self.image_format == "jpeg":
            pil_img.save(buf, format="JPEG", quality=self.quality)
        else:
            # These frames are transient, so spend as little time as possible
            # compressing them.
            pil_img.save(buf, format="PNG", compress_level=1)

        uri = f"data:{self._mime};base64," + base64.b64encode(buf.getvalue()).decode()
        with self._frame_lock:
            self._png = uri
            self._version += 1

    # -- keyboard input (replaces the pyglet window's key events) ------------
    def push_key(self, key):
        """Called from the server thread when a browser reports a keydown."""
        self._keys.put(key)

    def pop_key(self, timeout=None):
        """Next key pressed in a connected browser, or None. e.g. 'ArrowUp'."""
        try:
            return (
                self._keys.get(timeout=timeout) if timeout else self._keys.get_nowait()
            )
        except Empty:
            return None

    def process_events(self):
        pass

    def idle(self, seconds=0.00001):
        time.sleep(seconds)
