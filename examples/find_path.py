"""
This is the python script for question 1. In this script, you are required to implement a single agent path-finding algorithm
"""


#########################
# Your Task
#########################

# Run this script, required start and goal locations will be printed to terminal.
# Then replace this empty list with a list of location tuple, so that the agent move following the path from start location to goal location.
# For example : [(2,3),(2,4),(3,4)]
path = []



#########################
# Debugger and visualizer options
#########################

print("Start:  (6, 8)  Goal:  (3, 3)")

# Set these debug option to True if you want more information printed
debug = False
visualizer = True

# This function return a list of location tuple as the solution.
# @param start A tuple of (x,y) coordinates
# @param start_direction An Int indicate direction.
# @param goal A tuple of (x,y) coordinates
# @param env The flatland railway environment
# @return path A list of (x,y) tuple.
def get_path(start: tuple, start_direction: int, goal: tuple, env):
    return path


#########################
# You should not modify any codes below. You can read it know how we ran flatland environment.
########################

import glob, os

#import necessary modules that this python scripts need.
try:
    from flatland.envs.rail_env import RailEnv
    from controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator
except:
    print("Cannot load flatland modules! Make sure you activated flatland virtual environment with 'conda activate flatland-rl'")
    exit(1)

script_path = os.path.dirname(os.path.abspath(__file__))
test_cases = glob.glob(os.path.join(script_path,"test_0.pkl"))
evaluator(get_path,test_cases,debug,visualizer,1)


















