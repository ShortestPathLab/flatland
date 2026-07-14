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
import subprocess
import sys
import threading
import time
from queue import Empty, Queue

from flatland.utils import console
from flatland.utils.graphics_pil import PILSVG

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080

# How long to wait before deciding the native window came up. It dies almost
# immediately when it is going to fail (no display, no webview runtime), so this
# only needs to outlast process startup.
NATIVE_STARTUP_GRACE = 4.0

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

    code = (
        "import webview\n"
        f"webview.create_window({title!r}, {url!r}, width={size[0]}, height={size[1]})\n"
        "webview.start()\n"
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


class _Server:
    def __init__(self, host, port):
        # Imported here, on the calling thread, so a missing dependency surfaces
        # as a plain ImportError instead of a traceback buried in the server
        # thread followed by an opaque "failed to start".
        import nicegui  # noqa: F401

        self.host = host
        self.port = port
        self.started = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Don't let the sim race ahead of uvicorn's bind.
        if not self.started.wait(timeout=30):
            raise RuntimeError(
                f"flatland render server failed to start on {host}:{port}"
                + (
                    f": {self._error}"
                    if self._error
                    else " (is the port already in use?)"
                )
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

        def view(renderer, label=None, fit=True):
            if label:
                ui.label(label).classes("text-sm font-mono opacity-60")
            img = ui.interactive_image()
            # Fit inside the viewport, preserve aspect, and never scale beyond
            # natural size: upscaling a raster frame is what makes the art look
            # soft. Render at a higher cell_size / screen size for more detail.
            # 100dvh, not 100vh: on mobile the browser chrome makes vh taller
            # than what is actually visible, which would reintroduce the scroll.
            cap = "max-height:100dvh; max-width:100vw;" if fit else "max-width:100%;"
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

        @ui.page("/")
        def index():
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

        @ui.page("/env/{idx}")
        def single(idx: int):
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
            reconnect_timeout=10,
        )


def _ensure_server(host, port):
    global _server
    with _server_lock:
        if _server is None:
            _server = _Server(host, port)
        elif (host, port) != (_server.host, _server.port):
            # uvicorn is already bound; a second host/port in one process would
            # need a second server. Say so rather than silently ignoring it.
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
        wait_for_client=False,
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
        exit_on_close
            When a native window was opened, closing it ends the process
            (default True): a visualisation run is over once you stop watching
            it. Only applies to the native window - closing a *browser* tab
            never stops anything, since the tab may be one of several viewers on
            a remote machine. Set False to keep the process running after the
            window closes (e.g. if the render is a side-view onto a long job).

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
        self.port = int(port or os.environ.get("FLATLAND_RENDER_PORT", DEFAULT_PORT))
        self.wait_for_client_on_open = wait_for_client
        self.native = native and os.environ.get("FLATLAND_RENDER_NATIVE", "1") != "0"
        self.exit_on_close = exit_on_close
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

    # -- window lifecycle ----------------------------------------------------
    def open_window(self):
        assert self.window_open is False, "Window is already open!"
        server = _ensure_server(self.host, self.port)
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
            self._native_proc, reason = _try_open_native_window(url, size=size)
            opened_native = self._native_proc is not None
            if opened_native and self.exit_on_close:
                threading.Thread(
                    target=self._exit_when_window_closed,
                    args=(self._native_proc,),
                    daemon=True,
                ).start()

        # The panel stays live for the run: without a connection indicator,
        # "nothing is happening yet" and "you never opened the link" look
        # identical from the terminal, which is exactly where people get stuck.
        self._status = console.RenderStatus(
            url, server.port,
            get_viewers=lambda: server.clients,
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
        """End the process when the user closes the native window.

        A visualisation run is over once nobody is watching it. The main thread
        is typically blocked in the caller's step/render loop, so there is
        nothing to politely unwind to - hence the hard exit, mirroring what
        NiceGUI itself does. atexit handlers still run.
        """
        proc.wait()
        if self.closed:
            return  # close_window() killed it on purpose; not a user close.
        # Tear the live panel down first: os._exit would otherwise leave the
        # terminal with a hidden cursor and a half-drawn box.
        self._stop_status()
        console.info("Window closed - exiting.")
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

    # -- frame plumbing ------------------------------------------------------
    def latest_frame(self):
        """(version, data-uri) - called from the server thread."""
        with self._frame_lock:
            return self._version, self._png

    def has_viewers(self):
        return _server is not None and _server.clients > 0

    def show(self, block=False, from_event=False):
        if not self.window_open:
            self.open_window()
        if self.closed:
            return
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
