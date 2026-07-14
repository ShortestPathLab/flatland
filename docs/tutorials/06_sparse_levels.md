# Sparse levels and agent handling

- [Agents](#agents)
- [Level generation](#level-generation)
- [Observations](#observations)

## Agents
Agents are not permanent entities in the environment. An agent is removed from the environment as soon as it finishes its task. To keep interactions with the environment as simple as possible we do not modify the dimensions of the observation vectors nor the number of agents. Agents that have finished do not require any special treatment from the controller. Any action provided to these agents is simply ignored.

Start positions of agents are *not unique*. This means that many agents can start from the same position on the railway grid. It is important to keep in mind that whatever agent moves first will block the rest of the agents from moving into the same cell. Thus, the controller can already decide the ordering of the agents from the first step.

## Level Generation
Sparse levels are generated using the `sparse_rail_generator` and the `sparse_schedule_generator`.

### Rail Generation
The rail generation is done in a sequence of steps:
1. A number of city centers are placed in a grid of size `(height, width)`
2. Each city is connected to two neighbouring cities
3. Internal parallel tracks are generated in each city

### Schedule Generation
The `sparse_schedule_generator` produces tasks for the agents by selecting a starting city and a target city. The agent is then placed on an even track number in the starting city and faced such that a path exists to the target city. The task for the agent is to reach the target position as fast as possible.

## Observations
Have a look at [observations.py](https://github.com/ShortestPathLab/flatland/blob/master/flatland/envs/observations.py) or the documentation for more details on the observations.
