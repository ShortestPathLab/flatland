🚂 Flatland
========

![Flatland](https://i.imgur.com/0rnbSLY.gif)

<p style="text-align:center">
<img alt="repository" src="https://gitlab.aicrowd.com/flatland/flatland/badges/master/pipeline.svg">
<img alt="discord" src="https://gitlab.aicrowd.com/flatland/flatland/badges/master/coverage.svg">
</p>

Flatland is a open-source toolkit for developing and comparing Multi Agent Reinforcement Learning algorithms in little (or ridiculously large!) gridworlds.

[The official documentation](http://flatland.aicrowd.com/) contains full details about the environment and problem statement

Flatland is tested with Python 3.14 on modern versions of macOS, Linux and Windows. You may encounter problems with graphical rendering if you use WSL. Your [contribution is welcome](https://flatland.aicrowd.com/misc/contributing.html) if you can help with this!  

🏆 Challenges
---

This library was developed specifically for the AIcrowd [Flatland challenges](http://flatland.aicrowd.com/research/top-challenge-solutions.html) in which we strongly encourage you to take part in!

- [NeurIPS 2020 Challenge](https://www.aicrowd.com/challenges/neurips-2020-flatland-challenge/)
- [2019 Challenge](https://www.aicrowd.com/challenges/flatland-challenge)

📦 Setup
---

### Prerequisites

Flatland is developed as a [uv](https://docs.astral.sh/uv/) project. Install uv by following the
[official instructions](https://docs.astral.sh/uv/getting-started/installation/) — you do not need to install Python
yourself, as uv reads `.python-version` and provisions the right interpreter (currently 3.14.6) for you.

### Stable release

Install Flatland from PyPI:

```console
$ pip install flatland-rl
```

Or add it to an existing uv project:

```console
$ uv add flatland-rl
```

This is the preferred method to install Flatland, as it will always install the most recent stable release.

### Optional extras

The base install gives you everything needed to build, step and render an environment. The rest is opt-in:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `evaluator` | crowdai-api, redis, pandas, timeout-decorator, msgpack-numpy | `flatland.evaluators` and `flatland evaluate`, used to grade challenge submissions |
| `notebooks` | ipycanvas, ipython, ipywidgets | `flatland.utils.jupyter_utils` and `flatland.utils.editor`, the in-notebook helpers |
| `aws` | boto3 | S3 upload support in the evaluation service |

Install them individually or together:

```console
$ pip install 'flatland-rl[evaluator]'
$ pip install 'flatland-rl[evaluator,notebooks]'
```

### From sources

The Flatland code source is available from [AIcrowd gitlab](https://gitlab.aicrowd.com/flatland/flatland).

Clone the public repository:

```console
$ git clone git@gitlab.aicrowd.com:flatland/flatland.git
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

👥 Credits
---

This library was developed by [SBB](https://www.sbb.ch/en/), [Deutsche Bahn](https://www.deutschebahn.com/), [AIcrowd](https://www.aicrowd.com/) and [numerous contributors](http://flatland.aicrowd.com/misc/credits.html) and AIcrowd research fellows from the AIcrowd community. 

➕ Contributions
---
Please follow the [Contribution Guidelines](https://flatland.aicrowd.com/misc/contributing.html) for more details on how you can successfully contribute to the project. We enthusiastically look forward to your contributions!

💬 Communication
---

* [Discord Channel](https://discord.com/invite/hCR3CZG)
* [Discussion Forum](https://discourse.aicrowd.com/c/neurips-2020-flatland-challenge)
* [Issue Tracker](https://gitlab.aicrowd.com/flatland/flatland/issues/)

🔗 Partners
---

<a href="https://sbb.ch" target="_blank" style="margin-right:25px"><img src="https://i.imgur.com/OSCXtde.png" alt="SBB" width="200"/></a> 
<a href="https://www.deutschebahn.com/" target="_blank" style="margin-right:25px"><img src="https://i.imgur.com/pjTki15.png" alt="DB"  width="200"/></a>
<a href="https://www.aicrowd.com" target="_blank"><img src="https://avatars1.githubusercontent.com/u/44522764?s=200&v=4" alt="AICROWD"  width="200"/></a>
