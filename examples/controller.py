
from flatland.envs.rail_env import RailEnv
from flatland.evaluators.client import FlatlandRemoteClient
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_generators import complex_rail_generator, rail_from_file
from flatland.envs.schedule_generators import complex_schedule_generator, schedule_from_file
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from enum import IntEnum
import time, os, sys
import numpy as np

def eprint(*args, **kwargs):
    print("[ERROR] ",*args, file=sys.stderr, **kwargs)

output_template = "{0:15} | {1:15} | {2:15} | {3:15} | {4:15}"
output_header = output_template.format("Test case", "No. of agents","Agents done", "Sum of cost", "Makespan")


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


def path_controller(time_step,local_env: RailEnv, path_all: list, debug=False):
    action_dict = {}
    out_of_path = True
    inconsistent = False
    for agent_id in range(0, len(local_env.agents)):
        if time_step == 0:
            action_dict[agent_id] = Train_Actions.FORWARD
            out_of_path = False
        elif time_step >= len(path_all[agent_id]):
            action_dict[agent_id] = Train_Actions.NOTHING
        else:
            action_dict[agent_id] = get_action(agent_id, path_all[agent_id][time_step], local_env)
            if action_dict[agent_id] == -1:
                action_dict[agent_id] = 0
                if debug:
                    eprint("Agent {} cannot reach location {} from location {}. Path is inconsistent." \
                       .format(agent_id, path_all[agent_id][time_step], local_env.agents[agent_id].position))
                inconsistent = True

            out_of_path = False
    return action_dict,out_of_path,inconsistent

def get_action(agent_id: int, next_loc: tuple, env: RailEnv):
    current_loc = env.agents[agent_id].position
    current_direction = env.agents[agent_id].direction
    if  current_loc== next_loc:
        return Train_Actions.STOP

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
    elif move_direction - current_direction == 1 or move_direction - current_direction == -3 :
        return Train_Actions.RIGHT
    elif move_direction - current_direction == -1 or move_direction - current_direction == 3 :
        return Train_Actions.LEFT
    elif move_direction - current_direction == 2 or move_direction - current_direction == -2:
        return Train_Actions.FORWARD

    return -1

def check_conflict(time_step,path_all,local_env: RailEnv, debug=False):
    conflict = False
    for agent_id in range(0, len(local_env.agents)):
        if local_env.agents[agent_id].position != None and len(path_all[agent_id]) > time_step and local_env.agents[agent_id].position != path_all[agent_id][time_step]:
            conflict_id = -1
            cocurrent_move = False
            for i in range(0, len(local_env.agents)):
                if i != agent_id and  path_all[agent_id][time_step] == local_env.agents[i].position:
                    conflict_id = i
                    break
                if i != agent_id and time_step-1 >=0 and len(path_all[i]) > time_step-1 and path_all[agent_id][time_step] == path_all[i][time_step-1]:
                    conflict_id = i
                    cocurrent_move = True
                    break
            if debug:
                if conflict_id == -1:
                    eprint("Agent {} failed to move to {} at timestep {}. Check visualizer for what happened.".format(agent_id, path_all[agent_id][time_step], time_step))
                if cocurrent_move:
                    eprint("Agent {} failed to move to {} at timestep {} as lockstep movement with agent {} not allowed.".format(agent_id, path_all[agent_id][time_step], time_step,conflict_id))
                else:
                    eprint("Agent {} have conflict when trying to reach {} at timestep {} with Agent {}".format(agent_id, path_all[agent_id][time_step], time_step,conflict_id))
            conflict = True
    return conflict

def evaluator(get_path, test_cases: list, debug: bool, visualizer: bool,question_type: int):
    statistics = []
    print(output_header)
    for test_case in test_cases:
        test_name = os.path.basename(test_case)
        if debug:
            print("Loading evaluation: {}".format(test_case))
        local_env = RailEnv(width=1,
                            height=1,
                            rail_generator=rail_from_file(test_case),
                            schedule_generator=schedule_from_file(test_case),
                            remove_agents_at_target=True
                            # Removes agents at the end of their journey to make space for others
                            )

        local_env.reset()

        num_of_agents = local_env.get_num_agents()
        statistic_dict = {"test_case": test_name,"No. of agents":local_env.get_num_agents(), "time_step": 0, "num_done": 0, "sum_of_cost": 0, "done_percentage": 0,
                          "all_done": False}

        # Initiate the renderer
        if visualizer:
            env_renderer = RenderTool(local_env, gl="PILSVG",
                                      agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
                                      show_debug=debug,
                                      screen_height=800,  # Adjust these parameters to fit your resolution
                                      screen_width=800)  # Adjust these parameters to fit your resolution
            env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

        path_all = []
        if question_type == 1:
            agent_id = 0
            agent = local_env.agents[agent_id]
            path = get_path(agent.initial_position, agent.initial_direction, agent.target, local_env)
            path_all.append(path[:])
            if debug:
                print("Agent: {}, Path: {}".format(agent_id, path))
        elif question_type == 2:
            for agent_id in range(0, len(local_env.agents)):
                agent = local_env.agents[agent_id]
                path = get_path(agent.initial_position, agent.initial_direction, agent.target, local_env, agent_id,
                                path_all[:])
                path_all.append(path[:])
                if debug:
                    print("Agent: {}, Path: {}".format(agent_id, path))
        elif question_type == 3:
            path_all = get_path(local_env)
            if debug:
                for agent_id in range(0, len(local_env.agents)):
                    print("Agent: {}, Path: {}".format(agent_id, path_all[agent_id]))

        else:
            eprint("No such question type option.")
            exit(1)


        time_step = 0
        while time_step < local_env._max_episode_steps:

            action_dict, out_of_path, inconsistent = path_controller(time_step, local_env, path_all, debug)
            statistic_dict["time_step"] = time_step

            if inconsistent:
                if debug:
                    eprint("Can't finish test {} as path is inconsistent.".format(
                        test_case))
                    eprint("Press Enter to move to next test:")
                    input()
                break
            if out_of_path:
                if debug:
                    eprint("Reach last location in all paths. Current timestep: {}".format(time_step))
                    eprint("Can't finish test {}.".format(test_case))
                    eprint("Press Enter to move to next test:")
                    input()
                break

            # execuate action
            next_obs, all_rewards, done, _ = local_env.step(action_dict)

            if visualizer:
                env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

            num_done = 0
            new_cost = 0
            for agent_id in range(0, len(local_env.agents)):
                if done[agent_id]:
                    num_done += 1
                else:
                    new_cost += 1

            statistic_dict["num_done"] = num_done
            statistic_dict["done_percentage"] = round(num_done / len(local_env.agents), 2)
            statistic_dict["sum_of_cost"] += new_cost

            conflict = check_conflict(time_step, path_all, local_env, debug)
            if conflict:
                if debug:
                    eprint("Can't finish test {}. ".format(test_case))
                    eprint("Press Enter to move to next test:")
                    input()
                break


            if (done["__all__"]):
                statistic_dict["all_done"] = True
                if debug:
                    print("All agents reach destination at timestep: {}.  Move to next test in 1 seconds ...".format(time_step))
                time.sleep(1)
                break
            time_step += 1

        print(output_template.format(test_name, str(statistic_dict["No. of agents"]), str(statistic_dict["num_done"]),
                                     str(statistic_dict["sum_of_cost"]), str(statistic_dict["time_step"])))
        statistics.append(statistic_dict)


    count = 0
    sum_done_percent = 0
    sum_cost = 0
    num_done = 0
    sum_make=0
    sum_agents = 0
    for data in statistics:
        sum_done_percent += data["done_percentage"]
        sum_cost += data["sum_of_cost"]
        num_done += data["num_done"]
        sum_make += data["time_step"]
        sum_agents += data["No. of agents"]
        count+=1
    print(output_template.format("Summary", str(sum_agents)+" (sum)", str(num_done)+" (sum)",
                                 str(sum_cost)+" (sum)", str(sum_make)+" (sum)"))
    input("Press enter to exit:")

def remote_evaluator(get_path):
    remote_client = FlatlandRemoteClient()

    #####################################################################
    # Instantiate your custom Observation Builder
    #
    # You can build your own Observation Builder by following
    # the example here :
    # https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/envs/observations.py#L14
    #####################################################################
    my_observation_builder = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())


    #####################################################################
    # Main evaluation loop
    #
    # This iterates over an arbitrary number of env evaluations
    #####################################################################
    evaluation_number = 0
    while True:

        evaluation_number += 1
        # Switch to a new evaluation environemnt
        #
        # a remote_client.env_create is similar to instantiating a
        # RailEnv and then doing a env.reset()
        # hence it returns the first observation from the
        # env.reset()
        #
        # You can also pass your custom observation_builder object
        # to allow you to have as much control as you wish
        # over the observation of your choice.
        time_start = time.time()
        observation, info = remote_client.env_create(
            obs_builder_object=my_observation_builder
        )
        env_creation_time = time.time() - time_start
        if not observation:
            #
            # If the remote_client returns False on a `env_create` call,
            # then it basically means that your agent has already been
            # evaluated on all the required evaluation environments,
            # and hence its safe to break out of the main evaluation loop
            break

        print("Evaluation Number : {}".format(evaluation_number))

        #####################################################################
        # Access to a local copy of the environment
        #
        #####################################################################
        # Note: You can access a local copy of the environment
        # by using :
        #       remote_client.env
        #
        # But please ensure to not make any changes (or perform any action) on
        # the local copy of the env, as then it will diverge from
        # the state of the remote copy of the env, and the observations and
        # rewards, etc will behave unexpectedly
        #
        # You can however probe the local_env instance to get any information
        # you need from the environment. It is a valid RailEnv instance.
        local_env = remote_client.env
        number_of_agents = len(local_env.agents)

        # env_renderer = RenderTool(local_env, gl="PILSVG",
        #                       show_debug=True,
        #                       screen_height=1000,
        #                       screen_width=1000)

        # Now we enter into another infinite loop where we
        # compute the actions for all the individual steps in this episode
        # until the episode is `done`
        #
        # An episode is considered done when either all the agents have
        # reached their target destination
        # or when the number of time steps has exceed max_time_steps, which
        # is defined by :
        #
        # max_time_steps = int(4 * 2 * (env.width + env.height + 20))
        #

        path_all = get_path(local_env)

        time_taken_by_controller = []
        time_taken_per_step = []
        steps = 0
        while True:
            #####################################################################
            # Evaluation of a single episode
            #
            #####################################################################
            # Compute the action for this step by using the previously
            # defined controller
            time_start = time.time()
            action, out_of_path, inconsistent = path_controller(steps, local_env, path_all)
            if out_of_path: 
                action = {}
            
            time_taken = time.time() - time_start
            time_taken_by_controller.append(time_taken)

            # Perform the chosen action on the environment.
            # The action gets applied to both the local and the remote copy
            # of the environment instance, and the observation is what is
            # returned by the local copy of the env, and the rewards, and done and info
            # are returned by the remote copy of the env
            time_start = time.time()
            observation, all_rewards, done, info = remote_client.env_step(action)
            steps += 1
            time_taken = time.time() - time_start
            time_taken_per_step.append(time_taken)
            # env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

            if done['__all__']:
                print("Reward : ", sum(list(all_rewards.values())))
                #
                # When done['__all__'] == True, then the evaluation of this
                # particular Env instantiation is complete, and we can break out
                # of this loop, and move onto the next Env evaluation
                break

        np_time_taken_by_controller = np.array(time_taken_by_controller)
        np_time_taken_per_step = np.array(time_taken_per_step)
        print("=" * 100)
        print("=" * 100)
        print("Evaluation Number : ", evaluation_number)
        print("Current Env Path : ", remote_client.current_env_path)
        print("Env Creation Time : ", env_creation_time)
        print("Number of Steps : ", steps)
        print("Mean/Std of Time taken by Controller : ", np_time_taken_by_controller.mean(),
              np_time_taken_by_controller.std())
        print("Mean/Std of Time per Step : ", np_time_taken_per_step.mean(), np_time_taken_per_step.std())
        print("=" * 100)

    print("Evaluation of all environments complete...")
    ########################################################################
    # Submit your Results
    #
    # Please do not forget to include this call, as this triggers the
    # final computation of the score statistics, video generation, etc
    # and is necesaary to have your submission marked as successfully evaluated
    ########################################################################
    print(remote_client.submit())




