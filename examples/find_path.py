"""
This is the python script week 1 tutorial exercise.
"""


#########################
# Your Task
#########################

# Run this script, required start and goal locations will be printed to terminal.
# Then replace empty lists with lists of location tuple, so that each train can following the path to reach goal location.
# For example : [(2,3),(2,4),(3,4)]
train_1 = []
train_2 = []

# Turn debug to True to know what's wrong with your path.
debug = False



#########################
# You should not modify any codes below. You can read it know how we ran flatland environment.
########################

print("Train 1: Start  (6, 8)  Goal  (3, 3)")
print("Train 2: Start  (6, 0)  Goal  (0, 6)")

def get_path(local_env):
    return [train_1, train_2]

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
evaluator(get_path,test_cases,debug,True,3)


















