import runpy
import sys
from io import StringIO

from importlib.resources import as_file, files

from benchmarks.benchmark_utils import swap_attr

print("GRRRRRRRR run_all_examples.py")

for entry in [entry.name for entry in files('examples').iterdir() if
              entry.is_file()
              and entry.name.endswith(".py")
              and '__init__' not in entry.name
              and 'demo.py' not in entry.name
              and 'DELETE' not in entry.name
              ]:
    with as_file(files('examples').joinpath(entry)) as file_in:
        print("")
        print("")

        print("")
        print("*****************************************************************")
        print("Running {}".format(entry))
        print("*****************************************************************")
        with swap_attr(sys, "stdin", StringIO("q")):
            try:
                runpy.run_path(file_in, run_name="__main__", init_globals={
                    'argv': ['--sleep-for-animation=False']
                })
            except Exception as e:
                print(e)
            print("runpy done.")
        print("Done with {}".format(entry))
