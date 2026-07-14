#import necessary modules that this python scripts need.
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.utils.graphics_web import WEBGL
from flatland.utils.rendertools import RenderTool
import time


#Initialize Flatland Railway environment
railWaySettings = complex_rail_generator(
                   nr_start_goal=10, # Number of start location and train station pairs
                   nr_extra=10, # Extra rails between each pair of start and goal.
                   min_dist=10, # Minimum distance between start and goal locations
                   max_dist=99999,
                   seed=999)
 
env = RailEnv(  width=15, #Width/columns of grids
               height=15, #Height/rows of grids
               rail_generator= railWaySettings,
               number_of_agents=1) # How many trains are enabled.
env.reset() # initialize railway env
 
# Initiate the graphic module. This serves the env to a browser; open the URL it
# prints, and press the arrow keys in that page to drive the train.
render = RenderTool(env,
                         gl="WEB",  # only the web renderer reports key presses
                         show_debug=False,
                         screen_height=500,  # Render resolution height
                         screen_width=500,   # Render resolution width
                         wait_for_client=True)  # don't start until someone is watching
render.render_env(show=True, frames=False, show_observations=False)

# The web graphics layer is the one that collects key presses from the browser;
# the offline PIL layers draw images but have no keyboard.
def web_graphics(render_tool: RenderTool) -> WEBGL:
    gl = render_tool.gl
    if not isinstance(gl, WEBGL):
        raise RuntimeError("Keyboard control needs the web renderer, i.e. RenderTool(..., gl='WEB').")
    return gl

gl = web_graphics(render)

# Arrow key presses arrive from the browser page.
keyMap={"ArrowLeft":RailEnvActions.MOVE_LEFT,#Turn left
       "ArrowUp":RailEnvActions.MOVE_FORWARD,#Go forward
       "ArrowRight":RailEnvActions.MOVE_RIGHT,#Turn right
       "ArrowDown":RailEnvActions.STOP_MOVING#Stop
       }

#Define Controller
def my_controller(number_agents,first_makespan=False) -> dict[int, RailEnvActions]:
   _action: dict[int, RailEnvActions] = {}

   if first_makespan:
       #In the first frame of flatland, agents aren disabled,
       #  use any action to enable agents.
       for i in range(0,number_agents):
           _action[i] = RailEnvActions.MOVE_FORWARD
       return _action

   #Retrieve pressed action and put it into an action dictionary.
   #We only have 1 agent in this practice,
   #  thus we assign the pressed action only to agent 0
   pressedKey = gl.pop_key()  # most recent arrow key, or None
   _action[0] = keyMap.get(pressedKey, RailEnvActions.DO_NOTHING)
   return _action


   #Main loop
cost_dict={} #Record cost for each agent
makespan = -1 #all agents are disabled in the first makespan, so we start from -1..
all_done = False
while not all_done:
    time.sleep(0.25)
    #Call controller function to get the action dictionary for all agents
    _action = my_controller(len(env.agents), makespan == -1)
    #Pass action dictionary to railway env and execute all actions.
    obs, all_rewards, done, info = env.step(_action)
    all_done = done["__all__"]
 
   #count cost for each agent
    for agent_handle, value in done.items():
        if not value and agent_handle != "__all__":
            if agent_handle not in cost_dict.keys():
                cost_dict[agent_handle] = 1
            else:
                cost_dict[agent_handle] += 1
    makespan+=1
    render.render_env(show=True, frames=False, show_observations=False)
    time.sleep(0.15)
 
print("Sum of individual cost: ", sum(cost_dict.values()))
print("Total makespan: ", makespan)
