# -*- coding: utf-8 -*-

"""Command line interface for flatland.

    flatland demo        run a quick simulation and watch it
    flatland evaluate    run the grading service for challenge submissions

Heavy imports (the env, the renderer, the evaluator) are done inside the
commands rather than at module scope, so `flatland --help` stays instant.
"""

import time
from enum import Enum
from typing import Annotated, Optional

import typer

from flatland.utils import console

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Multi-agent reinforcement learning on trains.",
)


class ImageFormat(str, Enum):
    jpeg = "jpeg"
    png = "png"


def _version(show: bool):
    if show:
        import flatland

        typer.echo(f"flatland {flatland.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q",
                     help="Say nothing: no panels, no status display, no prompts."),
    ] = False,
    version: Annotated[
        Optional[bool],
        typer.Option("--version", callback=_version, is_eager=True,
                     help="Show the version and exit."),
    ] = None,
):
    console.set_quiet(quiet)


@app.command()
def demo(
    # -- the environment ----------------------------------------------------
    width: Annotated[int, typer.Option(help="Grid width, in cells.")] = 15,
    height: Annotated[int, typer.Option(help="Grid height, in cells.")] = 15,
    agents: Annotated[int, typer.Option(help="Number of trains.")] = 5,
    seed: Annotated[Optional[int], typer.Option(help="Seed, to make a run repeatable.")] = None,
    episodes: Annotated[
        int, typer.Option(help="Episodes to run. 0 runs until you stop it.")
    ] = 1,
    max_steps: Annotated[
        Optional[int], typer.Option(help="Cap the steps per episode.")
    ] = None,
    delay: Annotated[
        float, typer.Option(help="Seconds to pause between steps, so you can follow along.")
    ] = 0.3,
    # -- what you see -------------------------------------------------------
    headless: Annotated[
        bool,
        typer.Option("--headless",
                     help="Do not open a window; serve to a browser instead. "
                          "Use on a machine with no display."),
    ] = False,
    no_render: Annotated[
        bool,
        typer.Option("--no-render",
                     help="Do not visualise at all: no window, no server, no frames."),
    ] = False,
    host: Annotated[Optional[str], typer.Option(help="Address to serve on.")] = None,
    port: Annotated[Optional[int], typer.Option(help="Port to serve on.")] = None,
    cell_size: Annotated[
        Optional[int], typer.Option(help="Pixels per cell. Higher is sharper and slower.")
    ] = None,
    max_fps: Annotated[int, typer.Option(help="Cap on frames sent per second.")] = 30,
    image_format: Annotated[
        ImageFormat, typer.Option(help="Stream frames as JPEG (fast) or PNG (lossless).")
    ] = ImageFormat.jpeg,
    quality: Annotated[int, typer.Option(min=1, max=95, help="JPEG quality.")] = 90,
):
    """Run a simulation with randomly-acting trains, and watch it."""
    import numpy as np

    from flatland.envs.rail_env import RailEnv
    from flatland.envs.rail_generators import complex_rail_generator
    from flatland.envs.schedule_generators import complex_schedule_generator

    if no_render and headless:
        raise typer.BadParameter(
            "--no-render already means nothing is displayed; --headless has no effect "
            "alongside it. Use --headless on its own to watch in a browser."
        )
    if no_render and any(v is not None for v in (host, port, cell_size)):
        raise typer.BadParameter("--host/--port/--cell-size do nothing with --no-render.")

    rng = np.random.default_rng(seed)

    env = RailEnv(
        width=width,
        height=height,
        rail_generator=complex_rail_generator(
            nr_start_goal=10, nr_extra=1, min_dist=8, max_dist=99999,
            seed=seed if seed is not None else 1,
        ),
        schedule_generator=complex_schedule_generator(),
        number_of_agents=agents,
    )
    env._max_episode_steps = max_steps or int(15 * (env.width + env.height))

    renderer = None
    if not no_render:
        from flatland.utils.rendertools import RenderTool

        renderer = RenderTool(
            env,
            host=host,
            port=port,
            cell_size=cell_size,
            max_fps=max_fps,
            image_format=image_format.value,
            quality=quality,
            native=not headless,
        )

    episode = 0
    while episodes == 0 or episode < episodes:
        episode += 1
        env.reset()
        if renderer is not None:
            renderer.reset()

        done = {"__all__": False}
        steps = 0
        while not done["__all__"]:
            action = {i: int(rng.integers(0, 5)) for i, _ in enumerate(env.agents)}
            _, _, done, _ = env.step(action)
            steps += 1

            if renderer is not None:
                renderer.render_env(show=True, frames=False,
                                    show_observations=False, show_predictions=False)
                if delay:
                    time.sleep(delay)

        console.info(f"Episode {episode} finished after {steps} steps.")

    if renderer is not None:
        renderer.close_window()


@app.command()
def evaluate(
    tests: Annotated[
        str, typer.Option(exists=True, file_okay=False,
                          help="Folder containing the Flatland test environments."),
    ],
    service_id: Annotated[
        str, typer.Option(help="Must match the service id used by the client."),
    ] = "FLATLAND_RL_SERVICE_ID",
    shuffle: Annotated[
        bool, typer.Option(help="Shuffle the environments before evaluating."),
    ] = True,
    disable_timeouts: Annotated[
        bool, typer.Option(help="Turn off all evaluation timeouts."),
    ] = False,
    results_path: Annotated[
        Optional[str], typer.Option(help="Where to write the results metadata."),
    ] = None,
):
    """Run the grading service for challenge submissions."""
    # Imported lazily: the evaluation service is only available with the "evaluator" extra,
    # so that `flatland demo` and the env itself stay usable without it.
    try:
        import redis

        from flatland.evaluators.service import FlatlandRemoteEvaluationService
    except ImportError as e:
        console.warn(
            "The evaluator needs the optional 'evaluator' dependencies.\n"
            "Install them with:  pip install 'flatland-rl[evaluator]'\n"
            f"(missing: {e.name})",
            title="MISSING DEPENDENCIES",
        )
        raise typer.Exit(code=1)

    try:
        redis.Redis().ping()
    except redis.exceptions.ConnectionError:
        console.warn(
            "No Redis server is answering on localhost.\n"
            "The evaluator needs one running. Start it and try again.",
            title="REDIS NOT RUNNING",
        )
        raise typer.Exit(code=1)

    FlatlandRemoteEvaluationService(
        test_env_folder=tests,
        flatland_rl_service_id=service_id,
        visualize=False,
        result_output_path=results_path,
        verbose=False,
        shuffle=shuffle,
        disable_timeouts=disable_timeouts,
    ).run()


if __name__ == "__main__":
    app()
