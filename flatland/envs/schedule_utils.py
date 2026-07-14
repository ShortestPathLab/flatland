from typing import List, NamedTuple, Optional, Sequence

from flatland.core.grid.grid_utils import IntVector2DArray

Schedule = NamedTuple('Schedule', [('agent_positions', IntVector2DArray),
                                   # Sequence, not List: the generators build plain ints, while
                                   # from_agents_and_rail passes Grid4TransitionsEnum members.
                                   # List is invariant, so it could not accept both.
                                   ('agent_directions', Sequence[int]),
                                   ('agent_targets', IntVector2DArray),
                                   ('agent_speeds', List[float]),
                                   # Every schedule generator passes None; EnvAgent.from_schedule
                                   # falls back to a rate of 0 in that case.
                                   ('agent_malfunction_rates', Optional[List[int]]),
                                   ('max_episode_steps', int)])
