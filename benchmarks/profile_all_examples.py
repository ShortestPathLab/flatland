import cProfile
import runpy
import sys
from io import StringIO

from importlib.resources import as_file, files

from benchmarks.benchmark_utils import swap_attr


def profile(resource, entry):
    with as_file(files(resource).joinpath(entry)) as file_in:
        # TODO remove input() from examples
        print("*****************************************************************")
        print("Profiling {}".format(entry))
        print("*****************************************************************")
        with swap_attr(sys, "stdin", StringIO("q")):
            global my_func

            def my_func(): runpy.run_path(file_in, run_name="__main__")

            cProfile.run('my_func()', sort='time')


for entry in [entry.name for entry in files('examples').iterdir() if
              entry.is_file()
              and entry.name.endswith(".py")
              and '__init__' not in entry.name
              and 'demo.py' not in entry.name
              and 'DELETE' not in entry.name
              ]:
    profile('examples', entry)
