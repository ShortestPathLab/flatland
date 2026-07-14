"""Run every example, and fail if any of them raises.

This used to swallow exceptions (`except Exception as e: print(e)`), which meant CI stayed
green while examples rotted -- several were broken for a long time without anyone noticing.
A failing example is now a failing build.
"""
import runpy
import sys
import traceback
from io import StringIO

from importlib.resources import as_file, files

from benchmarks.benchmark_utils import swap_attr

# keyboard.py drives the env from browser arrow-key events and renders with
# wait_for_client=True, so it blocks forever without a human at a browser.
SKIP = ("__init__", "demo.py", "DELETE", "keyboard.py")

entries = sorted(
    entry.name
    for entry in files('examples').iterdir()
    if entry.is_file() and entry.name.endswith(".py") and not any(s in entry.name for s in SKIP)
)

failures = []

for entry in entries:
    with as_file(files('examples').joinpath(entry)) as file_in:
        print("")
        print("*****************************************************************")
        print("Running {}".format(entry))
        print("*****************************************************************")
        with swap_attr(sys, "stdin", StringIO("q")):
            try:
                runpy.run_path(file_in, run_name="__main__", init_globals={
                    'argv': ['--sleep-for-animation=False']
                })
            except Exception:
                traceback.print_exc()
                failures.append(entry)
        print("Done with {}".format(entry))

print("")
print("=================================================================")
print("{}/{} examples ok".format(len(entries) - len(failures), len(entries)))
if failures:
    print("FAILED: {}".format(", ".join(failures)))
    sys.exit(1)
