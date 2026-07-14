🚂 Getting Started with Flatland
========

This guide takes you from a fresh computer to a Flatland simulation running in a window on your
screen. It assumes **no prior experience** with terminals, Git or Python environments — if you have
used them before, skip ahead; nothing here is out of order.

Work through the steps in sequence. The whole thing takes about 15 minutes, most of which is waiting
for downloads.

- [Step 0: Open a terminal](#step-0-open-a-terminal)
- [Step 1: Install Git](#step-1-install-git)
- [Step 2: Install uv](#step-2-install-uv)
- [Step 3: Download Flatland](#step-3-download-flatland)
- [Step 4: Install Flatland](#step-4-install-flatland)
- [Step 5: Check it works](#step-5-check-it-works)
- [Step 6: Start your assignment](#step-6-start-your-assignment)
- [Troubleshooting](#troubleshooting)

Flatland is tested with Python 3.14 on modern versions of Windows, macOS and Linux, and inside WSL
(Windows Subsystem for Linux). Pick whichever of those you already have — all four are fully
supported, and the assignment is the same on each.

---

Step 0: Open a terminal
---

A **terminal** is a window where you type commands instead of clicking buttons. Nearly everything
below happens in one. Open it now and leave it open.

### Windows

Press <kbd>Win</kbd> + <kbd>X</kbd> and choose **Terminal** (on older machines: **Windows
PowerShell**). You can also press <kbd>Win</kbd>, type `terminal`, and press <kbd>Enter</kbd>.

You should see a prompt ending in `>`, something like:

```
PS C:\Users\you>
```

That is PowerShell, and it is what the Windows commands in this guide assume.

### WSL on Windows

WSL runs a real Ubuntu Linux system inside Windows. It is a good choice if your unit or your own
projects expect Linux tools, and Flatland fully supports it — including the graphical window.

If you do not have WSL yet, open **Terminal as Administrator** (press <kbd>Win</kbd>, type
`terminal`, then right-click it and choose **Run as administrator**) and run:

```console
> wsl --install
```

Restart your computer when it asks. On the next boot an Ubuntu window opens and asks you to invent a
username and password — the password is for Ubuntu, you will not see the characters as you type, and
that is normal.

From then on, open WSL by pressing <kbd>Win</kbd>, typing `ubuntu`, and pressing <kbd>Enter</kbd>.
Your prompt ends in `$`:

```
you@machine:~$
```

Once you are at that `$` prompt you are on Linux. **Follow the Linux instructions from here on**, not
the Windows ones.

### macOS

Press <kbd>Cmd</kbd> + <kbd>Space</kbd>, type `terminal`, press <kbd>Enter</kbd>. Your prompt ends in
`%` or `$`.

### Linux

Press <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>T</kbd>, or find **Terminal** in your applications
menu. Your prompt ends in `$`.

### How to read the commands in this guide

Code blocks show a prompt character that you **do not type**. Type only what comes after it.

```console
$ echo hello
hello
```

Here you type `echo hello` and press <kbd>Enter</kbd>; `hello` is the computer's reply. `$` marks a
macOS/Linux/WSL prompt and `>` marks a Windows PowerShell prompt.

---

Step 1: Install Git
---

**Git** is the tool that downloads the Flatland source code and tracks changes to your own work.

First, check whether you already have it:

```console
$ git --version
```

If that prints a version number (`git version 2.43.0` or similar), skip to
[Step 2](#step-2-install-uv). If it says *command not found* or *not recognized*, install it:

| Platform | Command |
| --- | --- |
| Windows | `winget install --id Git.Git -e` |
| WSL / Ubuntu / Debian | `sudo apt update && sudo apt install git` |
| macOS | `xcode-select --install` |
| Fedora | `sudo dnf install git` |
| Arch | `sudo pacman -S git` |

Notes:

- **Windows**: if `winget` is not recognised, download the installer from
  [git-scm.com](https://git-scm.com/download/win) and accept every default.
- **WSL/Linux**: `sudo` means "run as administrator". It will ask for the password you invented in
  Step 0. Nothing appears on screen as you type it — that is deliberate, just type and press
  <kbd>Enter</kbd>.
- **macOS**: this installs Apple's Command Line Tools, which include Git. A dialog box will pop up;
  click **Install** and wait.

**Close your terminal and open a new one**, then confirm it worked:

```console
$ git --version
git version 2.43.0
```

> **Why a new terminal?** A terminal reads the list of available commands once, when it starts. A
> program installed afterwards is invisible to it until you open a fresh one. If a command you just
> installed is "not found", this is almost always why.

---

Step 2: Install uv
---

**uv** manages Python for you. You do **not** need to install Python yourself, and you should not try
to — uv reads Flatland's `.python-version` file, downloads exactly the interpreter Flatland needs
(currently 3.14.6), and keeps it separate from anything else on your machine. This is what stops the
classic "it works on my laptop but not the tutor's" problem.

Run the installer for your platform:

**Windows** (PowerShell):

```console
> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS, Linux and WSL**:

```console
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Close your terminal and open a new one** (see the note in Step 1), then check:

```console
$ uv --version
uv 0.9.5
```

Any version number means you are fine. If it says *command not found*, see
[Troubleshooting](#uv-command-not-found).

---

Step 3: Download Flatland
---

Choose where to keep your work and move there. `cd` means "change directory":

```console
$ cd Documents
```

Now download the code. This creates a new `flatland` folder inside `Documents`:

```console
$ git clone https://github.com/ShortestPathLab/flatland.git
$ cd flatland
```

You are now *inside* the project folder, which is where the remaining commands must be run. If you
close your terminal and come back later, you will need to `cd` back here first.

> **WSL users**: keep your code in the Linux home directory (`~`, where you land by default), not in
> `/mnt/c/...`. Working across the Windows/Linux boundary is dramatically slower.

---

Step 4: Install Flatland
---

One command installs everything — the right Python, Flatland itself, and every dependency, all
pinned to the exact versions in `uv.lock`:

```console
$ uv sync
```

The first run downloads a lot and takes a few minutes. Later runs take seconds.

This creates a hidden `.venv` folder inside the project: a **virtual environment**, a private Python
installation used by this project alone. You never have to "activate" it — prefixing a command with
`uv run` (below) does that for you.

### The window: install the `native` extra

Flatland draws its simulations by running a small web server and displaying the frames it produces.
It can show those frames in two places:

- a **desktop window** that opens by itself, sits alongside your editor, and closes when the run
  ends — this is what you want;
- a **browser tab** you have to open by hand from a URL, and close by hand afterwards.

The desktop window is the `native` extra, and **we recommend it**. `uv sync` on a clone of this
repository installs it for you already, so if you followed Step 3 you have it. It matters when you
add Flatland to your *own* project — see [Step 6](#step-6-start-your-assignment).

Two platform notes:

- **WSL**: the window is a genuine Windows window, launched from Linux via Microsoft Edge in app
  mode. It works out of the box with no extra setup and no X server.
- **Linux desktop**: the window needs system GTK/WebKit libraries that pip cannot install. See
  [Troubleshooting](#the-window-does-not-open-on-linux) if it falls back to a browser tab.

---

Step 5: Check it works
---

Run the built-in demo — a small grid where five trains move at random:

```console
$ uv run flatland demo
```

A window should open showing trains moving on rails. When the episode finishes, the window closes and
the terminal prints how many steps it took. **If you see that, your setup is complete.**

`uv run` is how you run *everything* in this project: it finds the project's virtual environment,
makes sure it is up to date, and runs your command inside it. Use `uv run python my_script.py`, not
`python my_script.py`.

A few variations worth knowing:

```console
$ uv run flatland demo --agents 10 --width 30 --height 30   # a bigger, busier map
$ uv run flatland demo --seed 42                            # repeatable: same map every time
$ uv run flatland demo --delay 1                            # slow it down to one step per second
$ uv run flatland demo --headless                           # browser tab instead of a window
$ uv run flatland demo --help                               # every option
```

To confirm the whole library is healthy, run the test suite. It takes a few minutes:

```console
$ uv run pytest
```

---

Step 6: Start your assignment
---

The clone from Step 3 is the *library*. Your assignment is your own code, and it belongs in its own
project folder so that your work stays separate from Flatland's source.

If your unit gives you a starter repository, `git clone` that and follow its README — it will already
list Flatland as a dependency, and `uv sync` inside it is all you need.

If you are starting from scratch, create a project and add Flatland to it. Note the `[native]` — that
is the desktop window from Step 4, and this is the moment you have to ask for it:

```console
$ cd ..
$ uv init my-assignment
$ cd my-assignment
$ uv add "flatland-spl[native] @ git+https://github.com/ShortestPathLab/flatland"
```

> The package is published as **`flatland-spl`** (plain `flatland` on PyPI is an unrelated project),
> but the name you `import` in Python is still **`flatland`**.

Save this as `demo.py` in your new project to check everything is wired up:

```python
import time

import numpy as np

from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.rendertools import RenderTool

env = RailEnv(
    width=15,
    height=15,
    rail_generator=complex_rail_generator(
        nr_start_goal=10, nr_extra=1, min_dist=8, max_dist=99999, seed=1
    ),
    schedule_generator=complex_schedule_generator(),
    number_of_agents=5,
)
env.reset()

# native=True opens the desktop window; wait_for_client holds the first frame
# until the window is actually up, so you see the run from step 0.
renderer = RenderTool(env, native=True, wait_for_client=True)
renderer.reset()

# Random actions, purely to prove the setup works. Replacing this with a real
# planner is the assignment.
rng = np.random.default_rng(0)
done = {"__all__": False}
while not done["__all__"]:
    actions = {i: RailEnvActions(int(rng.integers(0, 5))) for i, _ in enumerate(env.agents)}
    _, _, done, _ = env.step(actions)
    renderer.render_env(show=True)
    time.sleep(0.2)

renderer.close_window()
```

Run it:

```console
$ uv run python demo.py
```

The trains move at random, so they mostly bump into each other and the episode ends after 240 steps.
Making them do something intelligent is the point of the assignment.

### Where to go next

- **[docs/tutorials/](docs/tutorials/)** — the real starting point. Building an environment, writing a
  custom observation builder and predictor, then stochastic and multi-speed trains.
- **[FAQ.md](FAQ.md)** — the environment, agent attributes and malfunctions.
- **[docs/specifications/](docs/specifications/)** — the railway model and the rendering, in depth.

---

Troubleshooting
---

### `uv`: command not found

You almost certainly need to open a new terminal — see the note at the end of
[Step 1](#step-1-install-git). If a fresh terminal still cannot find it, the installer put `uv`
somewhere your terminal does not look. Fix it for the current terminal with:

**macOS / Linux / WSL:**

```console
$ export PATH="$HOME/.local/bin:$PATH"
```

**Windows:**

```console
> $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
```

That lasts until you close the terminal. To make it permanent, run `uv --version` in a new terminal
after restarting your machine; if it still fails, ask on the unit forum rather than fighting it.

### `git clone` fails, or asks for a password

You are probably on a network that blocks Git. Try the clone again on a different connection (a phone
hotspot is a good test). Flatland is a public repository — you should never be asked for GitHub
credentials, so if you are, the URL has a typo.

### "The flatland render server cannot use port ..."

Flatland serves the visualisation over a local port, starting at 8080. If 8080 is busy — another dev
server, or a second Flatland running in the next terminal — it simply tries 8081, then 8082, and so
on, so you should never see this by default.

You only get this error if you *named* a port that something else already holds. That is treated as
an error rather than quietly moved, because asking for a specific port usually means something else
is expecting the render to be there. Either name a free one, or drop the option and let Flatland
choose:

```console
$ uv run flatland demo             # picks the first free port from 8080 up
$ uv run flatland demo --port 8099 # or insist on one
```

Whichever port it lands on is the one printed in the terminal panel and used by the window.

### The window does not open, and I get a URL instead

Flatland never fails hard here: if it cannot open a window, it serves the same simulation to your
browser and prints the address. Open that address (usually `http://127.0.0.1:8080/`) and you will see
exactly the same thing. Your setup is *working* — only the window is missing.

To get the window back, check you installed the `native` extra ([Step 4](#the-window-install-the-native-extra)).

### The window does not open on Linux

On a Linux desktop the window is drawn by `pywebview`, which needs GTK and WebKit system libraries
that pip cannot install for you. On Ubuntu or Debian:

```console
$ sudo apt install libgirepository1.0-dev gir1.2-webkit2-4.1 python3-gi
$ uv sync --reinstall
```

If that does not do it, use the browser tab — you lose nothing but convenience.

### The window does not open in WSL

The WSL window is a Windows Edge or Chrome window launched from Linux, so it needs one of those
installed on the **Windows** side. Both are standard on Windows 11 (Edge ships with the OS), so this shouldn't usually happen.

The one thing that will break it is binding the server to a specific address: Windows can only reach
the server through WSL's loopback forwarding, which covers the default wildcard bind. Do not pass
`--host 127.0.0.1`; leave `--host` alone and it will work.

### It was working, and now Python cannot find `flatland`

You are running `python` instead of `uv run python`, or you are in the wrong folder. Every command in
this project starts with `uv run`, and must be run from inside the project folder (the one containing
`pyproject.toml`). Check where you are with `pwd` (macOS/Linux/WSL) or `Get-Location` (Windows).

### Something else

Ask on the unit forum, or open an issue at
[github.com/ShortestPathLab/flatland/issues](https://github.com/ShortestPathLab/flatland/issues).
Include your operating system, the command you ran, and the **complete** error message — screenshots
of a single line are rarely enough to diagnose anything.
