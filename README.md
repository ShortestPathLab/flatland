🚂 Flatland
========

![Flatland](https://i.imgur.com/0rnbSLY.gif)

Flatland is an open-source toolkit for developing and comparing multi-agent path finding (MAPF) algorithms in little
(or ridiculously large!) gridworlds. Trains move on a grid of rails, each with its own start and target, and the hard
part is what happens when their paths conflict: rails restrict where a train may go next, trains block one another, and
they break down at inconvenient moments.

This repository is maintained by [ShortestPathLab](https://github.com/ShortestPathLab) and is used to teach
[FIT5222 Planning and Automated Reasoning](https://handbook.monash.edu/current/units/FIT5222) at Monash University.

Flatland is tested with Python 3.14 on modern versions of macOS, Linux and Windows, and inside WSL.

📦 Setup
---

New to this — or to Python, Git and the terminal generally? Follow the
[Getting Started guide](GETTING_STARTED.md), which walks through the setup from scratch on every
supported platform. The rest of this section is the short version.

### Prerequisites

Flatland is developed as a [uv](https://docs.astral.sh/uv/) project. Install uv by following the
[official instructions](https://docs.astral.sh/uv/getting-started/installation/) — you do not need to install Python
yourself, as uv reads `.python-version` and provisions the right interpreter (currently 3.14.6) for you.

### Install

Flatland is not published to a package index. Add it to an existing uv project straight from GitHub:

```console
$ uv add git+https://github.com/ShortestPathLab/flatland
```

The distribution is named `flatland-spl`, but the package you import is `flatland`:

```python
from flatland.envs.rail_env import RailEnv
```

### Optional extras

The base install gives you everything needed to build, step and render an environment. The rest is opt-in:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `native` | pywebview | Opening the renderer in a desktop window instead of a browser tab |
| `notebooks` | graphviz, ipycanvas, ipyevents, ipython, ipywidgets | `flatland.utils.jupyter_utils` and `flatland.utils.editor`, the in-notebook helpers |

```console
$ uv add "flatland-spl[notebooks] @ git+https://github.com/ShortestPathLab/flatland"
```

### From sources

Clone the repository:

```console
$ git clone https://github.com/ShortestPathLab/flatland.git
$ cd flatland
```

Once you have a copy of the source, create the virtual environment and install everything (including the development
dependencies) from the lockfile:

```console
$ uv sync
```

### Test installation

Test that the installation works:

```console
$ uv run flatland demo
```

You can also run the full test suite:

```console
$ uv run pytest
```

📖 Documentation
---

The [tutorials](docs/tutorials/) are the best starting point: they cover building an environment, writing a custom
observation builder and predictor, and the stochastic and multi-speed features. The [FAQ](FAQ.md) covers the
environment, the agent attributes and malfunctions, and [docs/specifications/](docs/specifications/) describes the
railway model, the rendering and the visualisation in more depth.

The docs are plain Markdown and reStructuredText, read directly from this repository — there is no separate site to
build.

➕ Contributions
---

Please follow the [Contribution Guidelines](CONTRIBUTING.rst) for more details on how you can successfully contribute
to the project. Issues and pull requests are welcome on
[GitHub](https://github.com/ShortestPathLab/flatland/issues).

This repository is maintained by Kevin Zheng (<kevin.zheng@monash.edu>). If you run into any problems, please open an
issue or get in touch.

📜 History and credits
---

Flatland began as a joint project of [SBB](https://www.sbb.ch/en/), [Deutsche Bahn](https://www.deutschebahn.com/) and
[AIcrowd](https://www.aicrowd.com/), built for a series of public multi-agent reinforcement learning challenges. This
repository is a fork of that original work, maintained independently by ShortestPathLab and repositioned around
multi-agent path finding for teaching and research. The reinforcement learning challenge scaffolding — the submission
and grading service in particular — has been removed, and the environment itself has been substantially reworked for
performance.

Thanks to the original authors and [numerous contributors](AUTHORS.md), whose work this builds on. Flatland is
distributed under the terms of the [LICENSE](LICENSE) it has always carried.
