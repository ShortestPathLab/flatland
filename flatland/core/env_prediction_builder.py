"""
PredictionBuilder objects are objects that can be passed to environments designed for customizability.
The PredictionBuilder-derived custom classes implement 2 functions, reset() and get([handle]).
If predictions are not required in every step or not for all agents, then

+ `reset()` is called after each environment reset, to allow for pre-computing relevant data.

+ `get()` is called whenever an step has to be computed, potentially for each agent independently in \
case of multi-agent environments.
"""
from typing import Generic, TypeVar, cast

from flatland.core.env import Environment

#: Environment the builder is attached to. Defaults to the generic :class:`Environment`, so
#: builders that do not care need not spell it out; the RailEnv-specific predictors subscript
#: it to get `self.env` typed as `RailEnv`.
#: Covariant so that a RailEnv-specific predictor is still accepted wherever a plain
#: PredictionBuilder is expected.
EnvType = TypeVar("EnvType", bound=Environment, covariant=True, default=Environment)


class PredictionBuilder(Generic[EnvType]):
    """
    PredictionBuilder base class.

    """

    #: Bound by :meth:`set_env` when the builder is attached to an environment.
    #: Reading it before then is a programming error.
    env: EnvType

    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth

    def set_env(self, env: Environment):
        # EnvType is covariant (so RailEnv-specific predictors remain usable wherever a plain
        # PredictionBuilder is expected), which rules it out as a parameter type. The env a
        # predictor is attached to is always the one it declared, so narrowing here is sound.
        self.env = cast(EnvType, env)

    def reset(self):
        """
        Called after each environment reset.
        """
        pass

    def get(self, handle: int = 0):
        """
        Called whenever get_many in the observation build is called.

        Parameters
        ----------
        handle : int, optional
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        function
            A prediction structure, specific to the corresponding environment.
        """
        raise NotImplementedError()
