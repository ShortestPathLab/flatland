from typing import Optional, Sequence, TypeAlias, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy import array

# A colour as accepted by the graphics layers: a name ("r", "gray"), an RGB(A)
# tuple of ints in 0..255, or a float sequence in 0..1 (eg what a matplotlib
# colormap returns). None means "use the default grid colour".
ColorSpec: TypeAlias = Union[str, Sequence[float], None]

# What adapt_color hands on to the drawing backend: an RGB(A) tuple of ints,
# or a colour name the backend understands.
Color: TypeAlias = Union[str, tuple[int, ...]]


class GraphicsLayer(object):
    """ Base class / interface for the graphics layers (PIL, PILSVG, WEB).
    """

    # Default colour for the grid lines. Concrete layers override it.
    tColGrid: tuple[int, ...]

    def __init__(self):
        pass

    def open_window(self):
        pass

    def close_window(self):
        pass

    def plot(self, *args, **kwargs):
        pass

    def scatter(self, *args, **kwargs):
        pass

    def text(self, *args, **kwargs):
        pass

    def prettify(self, *args, **kwargs):
        pass

    def show(self, block=False):
        pass

    def pause(self, seconds=0.00001):
        """ deprecated """
        pass

    def idle(self, seconds=0.00001):
        """ process any display events eg redraw, resize.
            Return only after the given number of seconds, ie idle / loop until that number.
        """
        pass

    def process_events(self):
        """ Pump whatever event loop the layer owns, if any. """
        pass

    def clf(self):
        pass

    def begin_frame(self):
        pass

    def endFrame(self):
        pass

    def clear_rails(self):
        """ Drop the (mostly static) rail layer, so that it is redrawn. """
        pass

    def get_image(self) -> Optional[np.ndarray]:
        pass

    def save_image(self, filename):
        pass

    def adapt_color(self, color: ColorSpec = None, lighten: bool = False) -> Color:
        adapted: Color
        if type(color) is str:
            if color == "red" or color == "r":
                adapted = (255, 0, 0)
            elif color == "gray":
                adapted = (128, 128, 128)
            else:
                adapted = color
        elif type(color) is list:
            adapted = tuple((array(color) * 255).astype(int))
        elif type(color) is tuple:
            if type(color[0]) is not int:
                gcolor = array(color)
                adapted = tuple((gcolor[:3] * 255).astype(int))
            else:
                adapted = tuple(int(iRGB) for iRGB in color)
        else:
            adapted = self.tColGrid

        # A colour name is passed straight through to the backend; there is
        # nothing to lighten.
        if lighten and not isinstance(adapted, str):
            adapted = tuple(int(255 - (255 - iRGB) / 3) for iRGB in adapted)

        return adapted

    def get_cmap(self, *args, **kwargs):
        return plt.get_cmap(*args, **kwargs)

    def set_rail_at(self, row, col, binary_trans, target=None, is_selected=False, rail_grid=None, num_agents=None,
                    show_debug=True):
        """ Set the rail at cell (row, col) to have transitions binary_trans.
            The target argument can contain the index of the agent to indicate
            that agent's target is at that cell, so that a station can be
            rendered in the static rail layer.
        """
        pass

    def set_agent_at(self, agent_idx, row, col, in_direction, out_direction, is_selected,
                     rail_grid=None, show_debug=False, clear_debug_text=True, malfunction=False):
        pass

    def set_cell_occupied(self, agent_idx, row, col):
        pass

    def resize(self, env):
        pass

    def build_background_map(self, dTargets):
        pass
