# Frequently Asked Questions (FAQs)

## Getting started

### How can I get started with Flatland?
Install Flatland by cloning the [repository](https://github.com/ShortestPathLab/flatland) and running `uv sync` in the
repository directory, or add it to an existing uv project with
`uv add git+https://github.com/ShortestPathLab/flatland`.

The [tutorials](docs/tutorials/) help you get a basic understanding of the flatland environment.

### What is an observation builder and which should I use?
Observation builders give you the possibility to generate custom observations for your planner. The observation builder
has access to all environment data and can perform any operations on it, as long as the data is not changed. The
[custom observations tutorial](docs/tutorials/02_observationbuilder.rst) will give you a sense of how to use them.

### What is a predictor and which one should I use?
Because railway traffic is limited to rails, many decisions that you have to take need to consider future situations and
detect upcoming conflicts ahead of time. Therefore, flatland provides the possibility of predictors that predict where
agents will be in the future. We provide a stock predictor that assumes each agent just travels along its shortest path.

You can build more elaborate predictors and use them as part of your observation builder. You find more information in
the [custom observations tutorial](docs/tutorials/02_observationbuilder.rst).

### What are rail and schedule generators?
To generate environments for Flatland you need to provide a railway infrastructure (rail) and a set of tasks for each
agent to complete (schedule).

## The environment

### What information is available about each agent?
Each agent is an object and contains the following information:

- `initial_position = attrib(type=Tuple[int, int])`: The initial position of an agent. This is where the agent will enter the environment. It is the start of the agent journey.
- `position = attrib(default=None, type=Optional[Tuple[int, int]])`: This is the actual position of the agent. It is updated every step of the environment. Before the agent has entered the environment and after it leaves the environment it is set to `None`
- `direction = attrib(type=Grid4TransitionsEnum)`: This is the direction an agent is facing. The values for directions are `North:0`, `East:1`, `South:2` and `West:3`.
- `target = attrib(type=Tuple[int, int])`: This is the target position the agent has to find and reach. Once the agent reaches this position its task is done.
- `moving = attrib(default=False, type=bool)`: Because agents can have malfunctions or be stopped because their path is blocked we store the current state of an agent. If `agent.moving == True` the agent is currently advancing. If it is `False` the agent is either blocked or broken.
- `speed_data = attrib(default=Factory(lambda: dict({'position_fraction': 0.0, 'speed': 1.0, 'transition_action_on_cellexit': 0})))`: This contains all the relevant information about the speed of an agent:
    - The attribute `'position_fraction'` indicates how far the agent has advanced within the cell. As soon as this value becomes larger than `1` the agent advances to the next cell as defined by `'transition_action_on_cellexit'`.
    - The attribute `'speed'` defines the travel speed of an agent. It can be any fraction smaller than 1.
    - The attribute `'transition_action_on_cellexit'` contains the information about the action that will be performed at the exit of the cell. Due to speeds smaller than 1. agents have to take several steps within a cell. We however only allow an action to be chosen at cell entry.
- `malfunction_data = attrib(default=Factory(lambda: dict({'malfunction': 0, 'malfunction_rate': 0, 'next_malfunction': 0, 'nr_malfunctions': 0,'moving_before_malfunction': False})))`: Contains all information relevant for agent malfunctions:
    - The attribute `'malfunction'` indicates if the agent is currently broken. If the value is larger than `0` the agent is broken. The integer value represents the number of `env.step()` calls the agent will still be broken.
    - The attribute `'nr_malfunctions'` is a counter that keeps track of the number of malfunctions a specific agent has had.
    - The attribute `'moving_before_malfunction'` is an internal parameter used to restart agents that were moving automatically after the malfunction is fixed.
- `status = attrib(default=RailAgentStatus.READY_TO_DEPART, type=RailAgentStatus)`: The status of the agent explains what the agent is currently doing. It can be in either one of these states:
    - `READY_TO_DEPART` not in grid yet (position is None)
    - `ACTIVE` in grid (position is not None), not done
    - `DONE` in grid (position is not None), but done
    - `DONE_REMOVED` removed from grid (position is None)

### What is the max number of timesteps per episode?
The maximum number of timesteps is `max_time_steps = 4 * 2 * (env.width + env.height + 20)`

### What are malfunctions and what can I do to resolve them?
Malfunctions occur according to a Poisson process. They hinder an agent from performing its actions and updating its
position. While an agent is malfunctioning it is blocking the paths for other agents. There is nothing you can do to fix
an agent, it will get fixed automatically as soon as `agent.malfunction_data['malfunction'] == 0`.

You can however adjust the other agent actions to avoid delay propagation within the railway network and keep traffic as
smooth as possible.

### Can agents communicate with each other?
There is no communication layer built into Flatland directly. You can however build a communication layer outside of the
Flatland environment if necessary.

### Can I use my own reward function?
Yes, you can shape the reward as you please. All information can be accessed directly in the env.

## Questions about bugs

### Why are my trains drawn outside of the rails?
If you render your environment and the agents appear to be off the rail it is usually due to changes in the railway
infrastructure. Make sure that you reset your renderer anytime the infrastructure changes by calling
`env_renderer.reset()`.
