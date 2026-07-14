from enum import IntEnum
from itertools import starmap
from typing import Any, Dict, List, Tuple, Optional, NamedTuple

from attr import attrs, attrib, Factory

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid_utils import IntVector2D
from flatland.envs.schedule_utils import Schedule


class RailAgentStatus(IntEnum):
    READY_TO_DEPART = 0  # not in grid yet (position is None) -> prediction as if it were at initial position
    ACTIVE = 1  # in grid (position is not None), not done -> prediction is remaining path
    DONE = 2  # in grid (position is not None), but done -> prediction is stay at target forever
    DONE_REMOVED = 3  # removed from grid (position is None) -> prediction is None


# speed_data and malfunction_data are heterogeneous: they hold floats, ints, bools and
# RailEnvActions, keyed by name.
AgentSpeedData = Dict[str, Any]
AgentMalfunctionData = Dict[str, Any]

Agent = NamedTuple('Agent', [('initial_position', IntVector2D),
                             ('initial_direction', Grid4TransitionsEnum),
                             ('direction', Grid4TransitionsEnum),
                             ('target', IntVector2D),
                             ('moving', bool),
                             ('speed_data', AgentSpeedData),
                             ('malfunction_data', AgentMalfunctionData),
                             ('handle', int),
                             ('status', RailAgentStatus),
                             ('position', Optional[IntVector2D]),
                             ('old_direction', Optional[Grid4TransitionsEnum]),
                             ('old_position', Optional[IntVector2D])])


@attrs
class EnvAgent:
    initial_position: IntVector2D = attrib()
    initial_direction: Grid4TransitionsEnum = attrib()
    direction: Grid4TransitionsEnum = attrib()
    target: IntVector2D = attrib()
    moving: bool = attrib(default=False)

    # speed_data: speed is added to position_fraction on each moving step, until position_fraction>=1.0,
    # after which 'transition_action_on_cellexit' is executed (equivalent to executing that action in the previous
    # cell if speed=1, as default)
    # N.B. we need to use factory since default arguments are not recreated on each call!
    speed_data: AgentSpeedData = attrib(
        default=Factory(lambda: dict({'position_fraction': 0.0, 'speed': 1.0, 'transition_action_on_cellexit': 0})))

    # if broken>0, the agent's actions are ignored for 'broken' steps
    # number of time the agent had to stop, since the last time it broke down
    malfunction_data: AgentMalfunctionData = attrib(
        default=Factory(
            lambda: dict({'malfunction': 0, 'malfunction_rate': 0, 'next_malfunction': 0, 'nr_malfunctions': 0,
                          'moving_before_malfunction': False})))

    handle: int = attrib(default=None)

    status: RailAgentStatus = attrib(default=RailAgentStatus.READY_TO_DEPART)
    # agents which have not departed (or which have been removed) have no position on the grid
    position: Optional[IntVector2D] = attrib(default=None)

    # used in rendering
    old_direction: Optional[Grid4TransitionsEnum] = attrib(default=None)
    old_position: Optional[IntVector2D] = attrib(default=None)

    deadline: Optional[int] = attrib(default=None)

    def reset(self):
        """
        Resets the agents to their initial values of the episode
        """
        self.position = None
        # TODO: set direction to None
        self.direction = self.initial_direction
        self.status = RailAgentStatus.READY_TO_DEPART
        self.old_position = None
        self.old_direction = None
        self.moving = False
        self.deadline = None

        # Reset agent values for speed
        self.speed_data['position_fraction'] = 0.
        self.speed_data['transition_action_on_cellexit'] = 0.

        # Reset agent malfunction values
        self.malfunction_data['malfunction'] = 0
        self.malfunction_data['nr_malfunctions'] = 0
        self.malfunction_data['moving_before_malfunction'] = False

    def to_agent(self) -> Agent:
        return Agent(initial_position=self.initial_position, initial_direction=self.initial_direction,
                     direction=self.direction, target=self.target, moving=self.moving, speed_data=self.speed_data,
                     malfunction_data=self.malfunction_data, handle=self.handle, status=self.status,
                     position=self.position, old_direction=self.old_direction, old_position=self.old_position)

    @classmethod
    def from_schedule(cls, schedule: Schedule) -> List["EnvAgent"]:
        """ Create a list of EnvAgent from lists of positions, directions and targets
        """
        speed_datas = []

        for i in range(len(schedule.agent_positions)):
            speed_datas.append({'position_fraction': 0.0,
                                'speed': schedule.agent_speeds[i] if schedule.agent_speeds is not None else 1.0,
                                'transition_action_on_cellexit': 0})

        malfunction_datas = []
        for i in range(len(schedule.agent_positions)):
            malfunction_datas.append({'malfunction': 0,
                                      'malfunction_rate': schedule.agent_malfunction_rates[
                                          i] if schedule.agent_malfunction_rates is not None else 0.,
                                      'next_malfunction': 0,
                                      'nr_malfunctions': 0})

        return list(starmap(EnvAgent, zip(schedule.agent_positions,
                                          schedule.agent_directions,
                                          schedule.agent_directions,
                                          schedule.agent_targets,
                                          [False] * len(schedule.agent_positions),
                                          speed_datas,
                                          malfunction_datas,
                                          range(len(schedule.agent_positions)))))

    @classmethod
    def load_legacy_static_agent(cls, static_agents_data: Tuple) -> List["EnvAgent"]:
        agents: List["EnvAgent"] = []
        for i, static_agent in enumerate(static_agents_data):
            agent = EnvAgent(initial_position=static_agent[0], initial_direction=static_agent[1],
                             direction=static_agent[1], target=static_agent[2], moving=static_agent[3],
                             speed_data=static_agent[4], malfunction_data=static_agent[5], handle=i)
            agents.append(agent)
        return agents
