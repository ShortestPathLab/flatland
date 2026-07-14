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

import base64
import io
import os
import threading
import time
from queue import Empty, Queue

from flatland.utils.graphics_pil import PILSVG

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080

# One server per process, shared by every RenderTool. Registering a route per
# renderer would mean mutating the route table after uvicorn has started; the
# pages below are static routes that read this registry at connection time.
_renderers = {}
_server = None
_server_lock = threading.Lock()


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
                + (f": {self._error}" if self._error else " (is the port already in use?)"))

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

        def view(renderer, label=None):
            if label:
                ui.label(label).classes("text-sm font-mono opacity-60")
            img = ui.interactive_image()
            # Fit the frame inside the window, preserving aspect, and never
            # scale it beyond its natural size: upscaling a raster frame is what
            # makes the art look soft. Render at a higher cell_size / screen
            # size instead if you want more detail.
            img.style("max-width:100%; max-height:100vh; width:auto; height:auto;"
                      " object-fit:contain; margin:0 auto;")
            img.props('fit=scale-down')
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
            for key in sorted(_renderers):
                view(_renderers[key], label=None if len(_renderers) == 1 else f"env {key}")

        @ui.page("/env/{idx}")
        def single(idx: int):
            renderer = _renderers.get(idx)
            if renderer is None:
                ui.label(f"No environment {idx}.")
                return
            view(renderer)

        app.on_startup(self.started.set)

        ui.run(host=self.host, port=self.port, show=False, reload=False,
               title="Flatland", reconnect_timeout=10)


def _ensure_server(host, port):
    global _server
    with _server_lock:
        if _server is None:
            _server = _Server(host, port)
        elif (host, port) != (_server.host, _server.port):
            # uvicorn is already bound; a second host/port in one process would
            # need a second server. Say so rather than silently ignoring it.
            print(f"[flatland] render server already running on "
                  f"{_server.host}:{_server.port}; ignoring {host}:{port}")
    return _server


class WEBGL(PILSVG):
    """Serve the rendered environment to a browser instead of a local window."""

    def __init__(self, width, height, jupyter=False, screen_width=800, screen_height=600,
                 cell_size=None, host=None, port=None, wait_for_client=False,
                 image_format="jpeg", quality=90, max_fps=30):
        """
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

    # -- window lifecycle ----------------------------------------------------
    def open_window(self):
        assert self.window_open is False, "Window is already open!"
        server = _ensure_server(self.host, self.port)
        _renderers[self._idx] = self
        self.window_open = True

        url = f"http://{server.host}:{server.port}/"
        if self.wait_for_client_on_open and server.clients == 0:
            print(f"[flatland] waiting for a browser at {url} ...", flush=True)
            if not server.wait_for_client(timeout=300):
                print("[flatland] no browser connected after 300s; continuing without one.")
        else:
            print(f"[flatland] rendering at {url}", flush=True)

    def close_window(self):
        _renderers.pop(self._idx, None)
        self.closed = True
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
            return self._keys.get(timeout=timeout) if timeout else self._keys.get_nowait()
        except Empty:
            return None

    def process_events(self):
        pass

    def idle(self, seconds=0.00001):
        time.sleep(seconds)
