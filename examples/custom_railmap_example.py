import random
from typing import Any, List

import numpy as np
from numpy.random import Generator

from flatland.core.grid.grid_utils import IntVector2D
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import RailGenerator, RailGeneratorProduct
from flatland.envs.schedule_generators import ScheduleGenerator
from flatland.envs.schedule_utils import Schedule
from flatland.utils.rendertools import RenderTool

random.seed(100)
np.random.seed(100)


def custom_rail_generator() -> RailGenerator:
    # A RailGenerator is called by RailEnv.reset() as
    #     rail_generator(width, height, num_agents, num_resets, np_random)
    # and returns the rail map plus an optional dict of hints for the schedule generator.
    def generator(width: int, height: int, num_agents: int, num_resets: int,
                  np_random: Generator) -> RailGeneratorProduct:
        rail_trans = RailEnvTransitions()
        grid_map = GridTransitionMap(width=width, height=height, transitions=rail_trans)
        rail_array = grid_map.grid
        rail_array.fill(0)
        new_tran = rail_trans.set_transition(1, 1, 1, 1)
        print(new_tran)
        rail_array[0, 0] = new_tran
        rail_array[0, 1] = new_tran
        return grid_map, None

    return generator


def custom_schedule_generator() -> ScheduleGenerator:
    # A ScheduleGenerator is called by RailEnv.reset() as
    #     schedule_generator(rail, num_agents, hints, num_resets, np_random)
    # This minimal example schedules no agents at all, so all the agent lists stay empty.
    def generator(rail: GridTransitionMap, num_agents: int, hints: Any, num_resets: int,
                  np_random: Generator) -> Schedule:
        agents_positions: List[IntVector2D] = []
        agents_direction: List[int] = []
        agents_target: List[IntVector2D] = []
        speeds: List[float] = []
        return Schedule(agent_positions=agents_positions, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=speeds, agent_malfunction_rates=None,
                        max_episode_steps=0)

    return generator


env = RailEnv(width=6, height=4, rail_generator=custom_rail_generator(), schedule_generator=custom_schedule_generator(),
              number_of_agents=1)

env.reset()

env_renderer = RenderTool(env)
env_renderer.render_env(show=True)

# uncomment to keep the renderer open
# input("Press Enter to continue...")
