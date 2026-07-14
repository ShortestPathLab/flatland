import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.rendertools import RenderTool

np.random.seed(1)

# Use the complex_rail_generator to generate feasible network configurations with corresponding tasks.
# Starting on simple small tasks is the best way to get familiar with the environment.

TreeObservation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())
env = RailEnv(width=20, height=20,
              rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=2, min_dist=8, max_dist=99999, seed=1),
              schedule_generator=complex_schedule_generator(), number_of_agents=3, obs_builder_object=TreeObservation)
env.reset()

env_renderer = RenderTool(env)


# Plug in your own planner here. As an example we pick actions uniformly at random.
class RandomPolicy:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        """
        :param state: input is the observation of the agent
        :return: returns an action
        """
        return np.random.choice(np.arange(self.action_size))


# Initialize the policy with the parameters corresponding to the environment and observation_builder
policy = RandomPolicy(218, 5)
n_episodes = 5

# Empty dictionary for all agent action
action_dict = dict()

for episode in range(1, n_episodes + 1):

    # Reset environment and get initial observations for all agents
    obs, info = env.reset()
    for idx in range(env.get_num_agents()):
        tmp_agent = env.agents[idx]
        tmp_agent.speed_data["speed"] = 1 / (idx + 1)
    env_renderer.reset()

    score = 0
    # Run episode
    for step in range(500):
        # Chose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            action = policy.act(obs[a])
            action_dict.update({a: action})
        # Environment step which returns the observations for all agents, their corresponding
        # reward and whether they are done
        next_obs, all_rewards, done, _ = env.step(action_dict)
        env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

        for a in range(env.get_num_agents()):
            score += all_rewards[a]
        obs = next_obs.copy()
        if done['__all__']:
            break
    print('Episode Nr. {}\t Score = {}'.format(episode, score))
