"""
ObservationBuilder objects are objects that can be passed to environments designed for customizability.
The ObservationBuilder-derived custom classes implement 2 functions, reset() and get() or get(handle).

+ `reset()` is called after each environment reset, to allow for pre-computing relevant data.

+ `get()` is called whenever an observation has to be computed, potentially for each agent independently in case of \
multi-agent environments.

"""
from typing import Dict, Generic, Optional, List, TypeVar, Union, cast

import numpy as np

from flatland.core.env import Environment

#: Type of the observation returned by an :class:`ObservationBuilder` for a single agent.
ObservationType = TypeVar("ObservationType")
#: Environment the builder is attached to. Defaults to the generic :class:`Environment`, so
#: builders that do not care (e.g. DummyObservationBuilder) need not spell it out; the
#: RailEnv-specific builders subscript it to get `self.env` typed as `RailEnv`.
#: Covariant so that a RailEnv-specific builder is still accepted wherever a plain
#: ObservationBuilder is expected (e.g. RailEnv's obs_builder_object parameter).
EnvType = TypeVar("EnvType", bound=Environment, covariant=True, default=Environment)


class ObservationBuilder(Generic[ObservationType, EnvType]):
    """
    ObservationBuilder base class.
    """

    #: Bound by :meth:`set_env` when the builder is attached to an environment.
    #: Reading it before then is a programming error.
    env: EnvType

    def set_env(self, env: Environment):
        # EnvType is covariant (so RailEnv-specific builders remain usable wherever a plain
        # ObservationBuilder is expected), which rules it out as a parameter type. The env a
        # builder is attached to is always the one it declared, so narrowing here is sound.
        self.env = cast(EnvType, env)

    def reset(self):
        """
        Called after each environment reset.
        """
        raise NotImplementedError()

    def get_many(self, handles: Optional[List[int]] = None) -> Union[Dict[int, ObservationType], ObservationType]:
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.

        Parameters
        ----------
        handles : list of handles, optional
            List with the handles of the agents for which to compute the observation vector.

        Returns
        -------
        function
            A dictionary of observation structures, specific to the corresponding environment, with handles from
            `handles` as keys. Builders which do not compute per-agent observations (such as
            :class:`DummyObservationBuilder`) may instead return a single observation for the whole environment.
        """
        observations: Dict[int, ObservationType] = {}
        if handles is None:
            handles = []
        for h in handles:
            observations[h] = self.get(h)
        return observations

    def get(self, handle: int = 0) -> ObservationType:
        """
        Called whenever an observation has to be computed for the `env` environment, possibly
        for each agent independently (agent id `handle`).

        Parameters
        ----------
        handle : int, optional
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        function
            An observation structure, specific to the corresponding environment.
        """
        raise NotImplementedError()

    def _get_one_hot_for_agent_direction(self, agent):
        """Retuns the agent's direction to one-hot encoding."""
        direction = np.zeros(4)
        direction[agent.direction] = 1
        return direction


class DummyObservationBuilder(ObservationBuilder[bool]):
    """
    DummyObservationBuilder class which returns dummy observations
    This is used in the evaluation service
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get_many(self, handles: Optional[List[int]] = None) -> bool:
        return True

    def get(self, handle: int = 0) -> bool:
        return True
