import argparse
import copy
import glob
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.malfunction_generators import (
    malfunction_from_file,
)
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.schedule_generators import (
    schedule_from_file,
)

if TYPE_CHECKING:
    from flatland.utils.rendertools import RenderTool

parser = argparse.ArgumentParser(description="Args for remote evaluation")
parser.add_argument(
    "--remote-mode",
    default=False,
    action="store_true",
    help="If running in remote mode",
)
parser.add_argument("--tests", type=str, default=None, help="Path for test cases")
parser.add_argument("-q", type=int, default=1, help="Question type")
parser.add_argument("-o", type=str, default=None, help="Output file")


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


_original_stdout = sys.stdout


def mute_print():
    _original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")


def unmute_print():
    sys.stdout.close()
    sys.stdout = _original_stdout


def eprint(*args, **kwargs):
    print("[ERROR] ", *args, file=sys.stderr, **kwargs)


def wprint(*args, **kwargs):
    print("[WARN] ", *args, file=sys.stderr, **kwargs)


def dprint(*args, **kwargs):
    print("[DEBUG] ", *args, **kwargs)


def _debug_pause(prompt="[DEBUG]  Press Enter to continue: "):
    """Pause so the user can read what was just printed, but never hang a
    non-interactive run (CI, piped stdin) on an input() nobody can answer."""
    if not sys.stdin.isatty():
        return
    try:
        input(prompt)
    except EOFError:
        pass


def _dprint_path(agent_id, path):
    if not path:
        wprint(
            "Agent {}: get_path() returned an empty path -- this agent will never depart.".format(
                agent_id
            )
        )
    else:
        dprint(
            "Agent {}: {} location(s), {} -> {}. Full path: {}".format(
                agent_id, len(path), path[0], path[-1], path
            )
        )


output_template = "{0:18} | {1:12} | {2:12} | {3:12} | {4:10} | {5:12} | {6:12} | {7:12} | {8:12} | {9:12}"
csv_template = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n"

output_header = output_template.format(
    "Test case",
    "Total agents",
    "Agents done",
    "DDLs met",
    "Plan Time",
    "SIC",
    "Makespan",
    "Penalty",
    "Final SIC",
    "P Score",
)


class Train_Actions(IntEnum):
    NOTHING = 0
    LEFT = 1
    FORWARD = 2
    RIGHT = 3
    STOP = 4


class Directions(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


def _shareable_path(path):
    """True if every location in path is a tuple of Python/numpy ints --
    immutable all the way down, so the tuples can be shared between copies.
    The set(map(type, ...)) passes stay at C speed; a per-element genexpr of
    isinstance checks was several times slower on long paths.
    """
    if set(map(type, path)) - {tuple}:
        return False
    inner_types = set(map(type, itertools.chain.from_iterable(path)))
    return all(t is int or issubclass(t, np.integer) for t in inner_types)


def fast_copy_paths(path_all):
    """Isolation copy of the paths handed to a student replan implementation.

    Equivalent to copy.deepcopy(path_all) for the documented path format
    (a list of paths, each a list of (x, y) integer location tuples): the
    location tuples are immutable, so they are shared instead of
    reconstructed. deepcopy rebuilt every tuple of every path on every
    replan call, which made it the single largest cost of an assessment
    run. Any path that does not match the documented format falls back to
    copy.deepcopy.
    """
    if type(path_all) is not list:
        return copy.deepcopy(path_all)
    copied = []
    for path in path_all:
        try:
            if type(path) is list and _shareable_path(path):
                copied.append(path[:])
                continue
        except TypeError:
            pass
        copied.append(copy.deepcopy(path))
    return copied


def path_controller(time_step, local_env: RailEnv, path_all: list, debug=False):
    action_dict = {}
    out_of_path = True
    inconsistent = False
    for agent_id in range(0, len(local_env.agents)):
        if time_step == 0:
            if len(path_all[agent_id]) > 0:
                action_dict[agent_id] = Train_Actions.FORWARD
            else:
                action_dict[agent_id] = Train_Actions.NOTHING
            out_of_path = False
        elif (
            time_step >= len(path_all[agent_id])
            or local_env.agents[agent_id].status == 3
        ):
            action_dict[agent_id] = Train_Actions.NOTHING
        else:
            action_dict[agent_id] = get_action(
                agent_id, path_all[agent_id][time_step], local_env
            )
            if action_dict[agent_id] == -1:
                action_dict[agent_id] = Train_Actions.STOP
                if debug:
                    eprint(
                        "Timestep {}: agent {} is at {} but its path says {} next, which is not reachable in one move. "
                        "Consecutive path locations must be adjacent grid cells -- the path is inconsistent, issuing STOP instead.".format(
                            time_step,
                            agent_id,
                            local_env.agents[agent_id].position,
                            path_all[agent_id][time_step],
                        )
                    )
                inconsistent = True

            out_of_path = False
    return action_dict, out_of_path, inconsistent


def get_action(agent_id: int, next_loc: Tuple[int, int], env: RailEnv) -> int:
    current_loc = env.agents[agent_id].position
    current_direction = env.agents[agent_id].direction
    if current_loc == next_loc:
        return Train_Actions.STOP

    assert (
        current_loc is not None
    ), "Agent {} is not on the grid, it has no position to move from.".format(agent_id)

    move_direction = 0
    if next_loc[0] - current_loc[0] == 1:
        move_direction = Directions.SOUTH
    elif next_loc[0] - current_loc[0] == -1:
        move_direction = Directions.NORTH
    elif next_loc[1] - current_loc[1] == -1:
        move_direction = Directions.WEST
    elif next_loc[1] - current_loc[1] == 1:
        move_direction = Directions.EAST
    else:
        move_direction = -1

    if move_direction == -1:
        return -1

    if move_direction == current_direction:
        return Train_Actions.FORWARD
    elif (
        move_direction - current_direction == 1
        or move_direction - current_direction == -3
    ):
        return Train_Actions.RIGHT
    elif (
        move_direction - current_direction == -1
        or move_direction - current_direction == 3
    ):
        return Train_Actions.LEFT
    elif (
        move_direction - current_direction == 2
        or move_direction - current_direction == -2
    ):
        return Train_Actions.FORWARD

    return -1


def check_conflict(time_step, path_all, local_env: RailEnv, debug=False):
    conflict = False
    failed_agents = []
    for agent_id in range(0, len(local_env.agents)):
        if (
            local_env.agents[agent_id].position is not None
            and len(path_all[agent_id]) > time_step
            and local_env.agents[agent_id].position != path_all[agent_id][time_step]
        ):
            conflict_id = -1
            failed_agents.append(agent_id)
            for i in range(0, len(local_env.agents)):
                if (
                    i != agent_id
                    and path_all[agent_id][time_step] == local_env.agents[i].position
                ):
                    conflict_id = i

            if debug:
                if conflict_id == -1:
                    wprint(
                        "Timestep {}: agent {} should be at {} but is at {} -- its move failed (blocked or malfunctioning). replan() will be called.".format(
                            time_step,
                            agent_id,
                            path_all[agent_id][time_step],
                            local_env.agents[agent_id].position,
                        )
                    )
                else:
                    wprint(
                        "Timestep {}: agent {} cannot enter {} -- agent {} is occupying that cell. replan() will be called.".format(
                            time_step,
                            agent_id,
                            path_all[agent_id][time_step],
                            conflict_id,
                        )
                    )
            conflict = True
    return conflict, failed_agents


@dataclass
class VisualiserOptions:
    """How to display an evaluation run. Defaults mirror ``flatland demo``.

    Pass an instance as ``evaluator(..., visualizer=VisualiserOptions(...))``;
    ``visualizer=True`` is shorthand for ``VisualiserOptions()`` and
    ``visualizer=False`` disables rendering entirely.
    """

    #: Seconds to pause after each timestep so the run is watchable.
    #: 0 runs at full speed.
    delay: float = 0.3
    #: Do not open a native window; print a URL and serve to a browser instead.
    #: Use on a machine with no display.
    headless: bool = False
    #: Hold the first frame until a viewer connects, so the run is seen from
    #: timestep 0. False starts immediately.
    wait: bool = True
    #: Address and port to serve on. Defaults: localhost, and the first free
    #: port from 8080 up.
    host: Optional[str] = None
    port: Optional[int] = None
    #: Pixels per grid cell. Default: scale the grid to fit the canvas.
    cell_size: Optional[int] = None
    #: Canvas size in pixels; the grid is scaled to fit. Defaults to the
    #: renderer's own choice (1600x1600 for the web renderer).
    screen_width: Optional[int] = None
    screen_height: Optional[int] = None
    #: Cap on frames streamed per second.
    max_fps: int = 30
    #: Stream frames as "jpeg" (fast) or "png" (lossless), and the JPEG
    #: quality (1-95).
    image_format: str = "jpeg"
    quality: int = 90
    #: Overlay agent IDs and targets on the rendering.
    show_debug: bool = True
    #: Closing the native window ends the whole process. Set False if the
    #: evaluation must keep running after the window is closed.
    exit_on_close: bool = True


#: One row of the per-test-case statistics table produced by :func:`evaluator`.
#: Declared with the functional syntax because some keys are not valid identifiers.
RunStatistics = TypedDict(
    "RunStatistics",
    {
        "test_case": str,
        "No. of agents": int,
        "time_step": int,
        "num_done": int,
        "deadlines_met": int,
        "sum_of_cost": int,
        "done_percentage": float,
        "all_done": bool,
        "cost": List[int],
        "penalty": List[int],
        "sic_final": int,
        "p": int,
        "f": float,
    },
)


def evaluator(
    get_path,
    test_cases: list,
    debug: bool = False,
    visualizer: Union[bool, VisualiserOptions] = False,
    question_type: int = 1,
    ddl: Optional[List[str]] = None,
    ddl_scale: float = 0.2,
    baseline_pscore: Dict[str, float] = {},
    save_pscore: Optional[str] = None,
    penalty_scale: float = 2,
    mute: bool = False,
    write: Optional[str] = None,
    replan: Optional[Callable[..., Any]] = None,
    max_steps: Optional[int] = None,
):
    """Run ``get_path`` against each test case and print a statistics table.

    get_path / replan
        The planner under evaluation. ``get_path``'s expected signature
        depends on ``question_type`` (1, 2 or 3); ``replan`` is only used for
        question type 3, where it is called whenever a malfunction or failed
        move is detected.
    debug
        Print a step-by-step account of the run: what was planned, which
        agents malfunctioned or were blocked, and why each episode ended.
        Pauses for Enter after problems when run from a terminal.
    visualizer
        ``False`` (default): run headless. ``True``: watch the run with the
        default :class:`VisualiserOptions`. A :class:`VisualiserOptions`
        instance: watch the run with custom settings (play speed, browser vs
        native window, ...).
    max_steps
        Cap the timesteps per episode. Default: each test case's own limit.
        Note that unfinished agents are scored at the episode limit, so
        lowering the cap changes scores.
    """
    if isinstance(visualizer, VisualiserOptions):
        vis_options: Optional[VisualiserOptions] = visualizer
    elif visualizer:
        vis_options = VisualiserOptions()
    else:
        vis_options = None

    statistics: List[RunStatistics] = []
    runtimes = []
    pscores = {}
    print(output_header, flush=True)
    out = open(write, "w+", 1) if write is not None else None

    # One renderer is live at a time, and it outlives its loop iteration on
    # purpose: it is retired when the NEXT test case starts, so the final
    # test's view is still up during the "Press enter to exit" pause below.
    env_renderer: Optional["RenderTool"] = None

    for i, test_case in enumerate(test_cases):
        test_name = (
            os.path.basename(os.path.dirname(test_case))
            + "/"
            + os.path.basename(test_case).replace(".pkl", "")
        )
        if debug:
            dprint("")
            dprint("=== Test {}/{}: {} ===".format(i + 1, len(test_cases), test_name))
            dprint("Loading {}".format(test_case))
        local_env = RailEnv(
            width=1,
            height=1,
            rail_generator=rail_from_file(test_case),
            schedule_generator=schedule_from_file(test_case),
            # schedule_generator=schedule_from_file(test_case, ddl_test_case),
            remove_agents_at_target=True,
            malfunction_generator_and_process_data=(
                malfunction_from_file(test_case) if question_type == 3 else None
            ),
            # Removes agents at the end of their journey to make space for others
            # This harness never reads the observations step() returns, and the
            # default GlobalObsForRailEnv builds full-grid arrays per agent per
            # step -- on large levels that was most of the episode runtime.
            obs_builder_object=DummyObservationBuilder(),
        )

        local_env.reset()

        # After the reset, never before it: reset() takes _max_episode_steps
        # from the schedule generator, so a cap set any earlier is silently
        # overwritten.
        if max_steps is not None:
            local_env._max_episode_steps = max_steps

        max_episode_steps = local_env._max_episode_steps
        assert (
            max_episode_steps is not None
        ), "the env must be reset before it can be evaluated."

        num_of_agents = local_env.get_num_agents()
        if debug:
            dprint(
                "{} agent(s) on a {}x{} grid; episode ends at timestep {}.".format(
                    num_of_agents, local_env.width, local_env.height, max_episode_steps
                )
            )
        statistic_dict: RunStatistics = {
            "test_case": test_name,
            "No. of agents": local_env.get_num_agents(),
            "time_step": 0,
            "num_done": 0,
            "deadlines_met": 0,
            "sum_of_cost": 0,
            "done_percentage": 0,
            "all_done": False,
            "cost": [0] * num_of_agents,
            "penalty": [0] * num_of_agents,
            # A scalar: sum_of_cost + sum(penalty), always written below before it is read.
            "sic_final": 0,
            "p": 0,
            "f": 0,
        }

        # Initiate the renderer
        if vis_options is not None:
            from flatland.utils.rendertools import RenderTool

            # Retire the previous test case's renderer first, so this one takes
            # its place in the single shared window rather than stacking under
            # it. This deregisters the old view; the window itself is opened
            # once per process and lives for the whole run.
            if env_renderer is not None:
                env_renderer.close_window()

            env_renderer = RenderTool(
                local_env,
                show_debug=vis_options.show_debug,
                screen_width=vis_options.screen_width,
                screen_height=vis_options.screen_height,
                cell_size=vis_options.cell_size,
                host=vis_options.host,
                port=vis_options.port,
                native=not vis_options.headless,
                wait_for_client=vis_options.wait,
                exit_on_close=vis_options.exit_on_close,
                image_format=vis_options.image_format,
                quality=vis_options.quality,
                max_fps=vis_options.max_fps,
            )
            env_renderer.render_env(
                show=True, show_observations=False, show_predictions=False
            )
        path_all = []
        if debug:
            dprint("Planning initial paths with get_path() ...")
        start_t = time.time()
        if mute:
            mute_print()
        if question_type == 1:
            agent_id = 0
            agent = local_env.agents[agent_id]
            path = get_path(
                agent.initial_position,
                agent.initial_direction,
                agent.target,
                copy.deepcopy(local_env.rail),
                local_env._max_episode_steps,
            )
            path_all.append(path[:])
            if debug:
                _dprint_path(agent_id, path)
        elif question_type == 2:
            for agent_id in range(0, len(local_env.agents)):
                agent = local_env.agents[agent_id]
                path = get_path(
                    agent.initial_position,
                    agent.initial_direction,
                    agent.target,
                    copy.deepcopy(local_env.rail),
                    agent_id,
                    path_all[:],
                    local_env._max_episode_steps,
                )
                path_all.append(path[:])
                if debug:
                    _dprint_path(agent_id, path)
        elif question_type == 3:
            if ddl:
                deadlines = local_env.read_deadlines(ddl[i])
            else:
                expected_delay = (
                    local_env.malfunction_process_data.malfunction_rate
                    * (local_env.width + local_env.height)
                    * (
                        (
                            local_env.malfunction_process_data.min_duration
                            + local_env.malfunction_process_data.max_duration
                        )
                        / 2
                    )
                )
                deadlines = local_env.generate_deadlines(
                    ddl_scale,
                    group_size=max(1, len(local_env.agents) // 5),
                    malfunction_scale=(
                        1 + expected_delay / (local_env.width + local_env.height) / 2
                    ),
                )
                local_env.save_deadlines(test_case[:-4], deadlines)
            local_env.set_deadlines(deadlines)

            path_all = get_path(
                copy.deepcopy(local_env.agents),
                copy.deepcopy(local_env.rail),
                local_env._max_episode_steps,
            )
            if debug:
                if len(path_all) != num_of_agents:
                    wprint(
                        "get_path() returned {} path(s) for {} agents.".format(
                            len(path_all), num_of_agents
                        )
                    )
                for agent_id in range(0, min(len(path_all), num_of_agents)):
                    _dprint_path(agent_id, path_all[agent_id])

        else:
            eprint("No such question type option.")
            exit(1)
        if mute:
            unmute_print()
        runtimes.append(round(time.time() - start_t, 2))
        if debug:
            dprint("Planning finished in {}s. Running the episode ...".format(runtimes[-1]))

        replan_runtime = 0
        time_step = 0
        out_of_path = False
        inconsistent = False
        done_agents: set = set()
        # `step()` returns the very same dict, so this is the state before the first step
        done = local_env.dones
        while time_step < max_episode_steps:
            if out_of_path:
                if debug:
                    wprint(
                        "Timestep {}: every path is exhausted, but only {}/{} agents reached their target.".format(
                            time_step, statistic_dict["num_done"], num_of_agents
                        )
                    )
                    wprint(
                        "The paths are too short to finish test {}. Ending this test.".format(
                            test_name
                        )
                    )
                    _debug_pause("[DEBUG]  Press Enter to move to the next test: ")
                break

            if inconsistent and debug:
                _debug_pause()

            action_dict, out_of_path, inconsistent = path_controller(
                time_step, local_env, path_all, debug
            )
            statistic_dict["time_step"] = time_step

            malfunction_before = [
                agent.malfunction_data["malfunction"] > 0 and agent.status < 2
                for agent in local_env.agents
            ]
            # execuate action
            next_obs, all_rewards, done, _ = local_env.step(action_dict)

            if env_renderer is not None:
                env_renderer.render_env(
                    show=True, show_observations=False, show_predictions=False
                )
                if vis_options is not None and vis_options.delay:
                    time.sleep(vis_options.delay)

            # Find malfunctioning and failed-execution agents. Then call the replan function.
            if question_type == 3:
                conflict = False
                failed_agents = []
                new_malfunctions = []
                if time_step != 0:
                    conflict, failed_agents = check_conflict(
                        time_step, path_all, local_env, debug
                    )
                    if conflict and debug:
                        _debug_pause()
                for agent in local_env.agents:
                    if agent.status != 3 and time_step >= len(path_all[agent.handle]):
                        failed_agents.append(agent.handle)
                    if (
                        agent.malfunction_data["malfunction"] > 0
                        and agent.status < 2
                        and not malfunction_before[agent.handle]
                    ):
                        new_malfunctions.append(agent.handle)
                if debug:
                    for handle in new_malfunctions:
                        broken = local_env.agents[handle]
                        wprint(
                            "Timestep {}: agent {} malfunctioned at {} -- immobilised for the next {} timestep(s).".format(
                                time_step,
                                handle,
                                broken.position,
                                broken.malfunction_data["malfunction"],
                            )
                        )
                if len(new_malfunctions) != 0 or conflict:
                    if debug:
                        dprint(
                            "Timestep {}: calling replan() (malfunctioning agents: {}; off-plan or out-of-path agents: {}) ...".format(
                                time_step, new_malfunctions, failed_agents
                            )
                        )
                    replan_start = time.time()
                    if mute:
                        mute_print()
                    assert (
                        replan is not None
                    ), "a replan function is required for question type 3."
                    new_paths = replan(
                        copy.deepcopy(local_env.agents),
                        copy.deepcopy(local_env.rail),
                        time_step,
                        fast_copy_paths(path_all),
                        local_env._max_episode_steps,
                        new_malfunctions,
                        failed_agents,
                    )
                    if mute:
                        unmute_print()
                    replan_time = round(time.time() - replan_start, 2)
                    replan_runtime += replan_time
                    path_all = new_paths
                    if debug:
                        dprint("replan() returned in {}s.".format(replan_time))

            num_done = 0
            num_deadlines_met = 0
            newly_done = []
            for agent_id in range(0, len(local_env.agents)):
                deadline: Optional[int] = local_env.agents[agent_id].deadline
                if local_env.agents[agent_id].status in [2, 3]:
                    num_done += 1
                    if agent_id not in done_agents:
                        done_agents.add(agent_id)
                        newly_done.append(agent_id)
                    if question_type == 3 and deadline:
                        if statistic_dict["cost"][agent_id] <= deadline:
                            num_deadlines_met += 1
                    else:
                        num_deadlines_met += 1
                else:
                    if (
                        question_type == 3
                        and deadline is not None
                        and time_step > deadline
                    ):
                        statistic_dict["penalty"][agent_id] += 1
                    statistic_dict["cost"][agent_id] += 1

            statistic_dict["num_done"] = num_done
            statistic_dict["done_percentage"] = round(
                num_done / len(local_env.agents), 2
            )
            statistic_dict["deadlines_met"] = num_deadlines_met

            if debug and newly_done:
                dprint(
                    "Timestep {}: agent(s) {} reached their target ({}/{} done).".format(
                        time_step, newly_done, num_done, num_of_agents
                    )
                )

            if done["__all__"]:
                statistic_dict["all_done"] = True
                if debug:
                    dprint(
                        "Timestep {}: all {} agents reached their targets. Moving to the next test.".format(
                            time_step, num_of_agents
                        )
                    )
                break
            time_step += 1

        if debug and not statistic_dict["all_done"] and not out_of_path:
            dprint(
                "Timestep limit ({}) reached with {}/{} agents at their target. Moving to the next test.".format(
                    max_episode_steps, statistic_dict["num_done"], num_of_agents
                )
            )

        runtimes[-1] += replan_runtime
        # End of one episode.
        max_episode_steps = local_env._max_episode_steps
        assert max_episode_steps is not None, "the env is reset above, so this is set"
        for agent_id in range(0, len(local_env.agents)):
            if done[agent_id]:
                statistic_dict["sum_of_cost"] += statistic_dict["cost"][agent_id]
            else:
                statistic_dict["sum_of_cost"] += max_episode_steps
        statistic_dict["sic_final"] = statistic_dict["sum_of_cost"] + sum(
            statistic_dict["penalty"]
        )
        if question_type == 1:
            statistic_dict["p"] = int(statistic_dict["sic_final"] / num_of_agents)
        else:
            statistic_dict["p"] = int(statistic_dict["sic_final"] / num_of_agents)
            if baseline_pscore:
                statistic_dict["f"] = min(
                    round(baseline_pscore[test_case] / statistic_dict["p"], 2), 1.0
                )
        pscores[test_case] = statistic_dict["p"]
        print(
            output_template.format(
                test_name,
                str(statistic_dict["No. of agents"]),
                str(statistic_dict["num_done"]),
                str(statistic_dict["deadlines_met"]),
                str(runtimes[-1]),
                str(statistic_dict["sum_of_cost"]),
                str(statistic_dict["time_step"]),
                str(sum(statistic_dict["penalty"])),
                str(statistic_dict["sic_final"]),
                str(statistic_dict["p"])
                + ("({})".format(statistic_dict["f"]) if baseline_pscore else ""),
            ),
            flush=True,
        )
        if out is not None:
            out.write(
                csv_template.format(
                    test_name,
                    str(statistic_dict["No. of agents"]),
                    str(statistic_dict["num_done"]),
                    str(statistic_dict["deadlines_met"]),
                    str(runtimes[-1]),
                    str(statistic_dict["sum_of_cost"]),
                    str(statistic_dict["time_step"]),
                    str(sum(statistic_dict["penalty"])),
                    str(statistic_dict["sic_final"]),
                    str(statistic_dict["p"])
                    + ("({})".format(statistic_dict["f"]) if baseline_pscore else ""),
                )
            )
        statistics.append(statistic_dict)

    count = 0
    sum_done_percent = 0
    sum_cost = 0
    num_done = 0
    sum_make = 0
    sum_agents = 0
    sum_penalty = 0
    sum_sic_final = 0
    sum_p = None
    sum_f = 0
    sum_runtime = round(sum(runtimes), 2)
    sum_ddl_met = 0
    for data in statistics:
        sum_done_percent += data["done_percentage"]
        sum_cost += data["sum_of_cost"]
        num_done += data["num_done"]
        sum_make += data["time_step"]
        sum_agents += data["No. of agents"]
        sum_penalty += sum(data["penalty"])
        sum_sic_final += data["sic_final"]
        sum_ddl_met += data["deadlines_met"]
        sum_f += data["f"]
        count += 1
    if question_type == 1:
        sum_p = int(sum_cost / sum_agents)
        pscores["q1"] = sum_p
        if baseline_pscore:
            sum_f = max(round(baseline_pscore["q1"] / sum_p, 2), 1.0)
    if save_pscore:
        with open(save_pscore, "w+") as f:
            f.write(json.dumps(pscores))
    print(
        output_template.format(
            "Summary",
            str(sum_agents) + " (sum)",
            str(num_done) + " (sum)",
            str(sum_ddl_met) + "(sum)",
            str(sum_runtime) + "(sum)",
            str(sum_cost) + " (sum)",
            str(sum_make) + " (sum)",
            str(sum_penalty) + " (sum)",
            str(sum_sic_final) + " (sum)",
            str(sum_p) + " (final)" + (str(sum_f) if baseline_pscore else ""),
        ),
        flush=True,
    )
    if out is not None:
        out.write(
            csv_template.format(
                "Summary",
                str(sum_agents) + " (sum)",
                str(num_done) + " (sum)",
                str(sum_ddl_met) + "(sum)",
                str(sum_runtime) + "(sum)",
                str(sum_cost) + " (sum)",
                str(sum_make) + " (sum)",
                str(sum_penalty) + " (sum)",
                str(sum_sic_final) + " (sum)",
                str(sum_p) + " (final)" + (str(sum_f) if baseline_pscore else ""),
            )
        )
        out.close()
    # Hold the process open so the rendered view stays up until the user is done looking at it.
    # Only meaningful with a human at a terminal: under CI, a pipe, or a redirected stdin there is
    # nobody to press enter. isatty() covers the usual non-interactive cases, but a stdin can also
    # claim to be a terminal and still be at EOF (redirecting from /dev/null under MSYS does
    # exactly that), so swallow the EOFError too rather than dying on the last line of a long run.
    if not mute and sys.stdin.isatty():
        try:
            input("Press enter to exit:")
        except EOFError:
            pass


def remote_evaluator(get_path, args, replan=None):
    args = parser.parse_args(args[1:])
    path = args.tests
    q = args.q
    tests = glob.glob("{}/level_*/test_*.pkl".format(path))
    tests.sort()
    if q == 1:
        evaluator(get_path, tests, False, False, 1, mute=True, write=args.o)
    elif q == 2:
        evaluator(get_path, tests, False, False, 2, mute=True, write=args.o)
    elif q == 3:
        deadline_files = [test.replace(".pkl", ".ddl") for test in tests]
        evaluator(
            get_path,
            tests,
            False,
            False,
            3,
            deadline_files,
            penalty_scale=4,
            mute=True,
            write=args.o,
            replan=replan,
        )
