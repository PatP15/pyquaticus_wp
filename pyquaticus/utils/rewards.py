# (C) 2021 Massachusetts Institute of Technology.

# Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

# The software/firmware is provided to you on an As-Is basis

# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically

"""
#Configureable Rewards
params{
    # -- NOTE --
    #   All bearings are in nautical format
    #                 0
    #                 |
    #          270 -- . -- 90
    #                 |
    #                180
    #
    # This can be converted the standard bearing format that is counterclockwise
    # using the heading_angle_conversion(deg) function found in utils.py
    #
    #
    #
    #
    #
    "num_players": int, #Number of players currently in the game
    "num_teammates": int, #Number of teammates currently in the game
    "num_opponents": int, #Number of opponents currently in the game
    "agent_id": int, #ID of agent rewards are being computed for
    "capture_radius": int, #The radius required to grab, capture a flag; and tag opponents
    "team_flag_pickup": bool,    # Indicates if team grabs flag
    "team_flag_capture": bool,   # Indicates if team captures flag
    "opponent_flag_pickup": bool, # Indicates if opponent grabs flag 
    "opponent_flag_capture": bool, #Indicates if opponent grabs flag
    "team_flag_bearing": float, # Agents bearing to team flag
    "team_flag_distance": float, # Agents distance to team flag
    "opponent_flag_bearing": float, # Agents bearing to opponents flag
    "opponent_flag_distance": float, #Agents distance to opponents flag
    "speed": float, #Agents current speed
    "tagging_cooldown": bool, # True when agents tag is on cooldown False otherwise
    "thrust": float, # Agents current thrust
    "has_flag": bool, #Indicates if agent currently has opponents flag
    "on_own_side": bool, #Indicates if agent is currently on teams side
    "heading": float, #Agents yaw in degrees
    "wall_0_bearing": float, #Agents bearing towards 
    "wall_0_distance": float, #Agents distance towards
    "wall_1_bearing": float, #Agents bearing towards 
    "wall_1_distance": float, #Agents distance towards
    "wall_2_bearing": float, #Agents bearing towards
    "wall_2_distance": float, #Agents distance towards
    "wall_3_bearing": float, #Agents bearing towards
    "wall_3_distance": float, #Agents distance towards
    "wall_distances": Dict, (wallid, distance)
    "agent_captures": list, #List of agents that agent has tagged 0 not tagged 1 tagged by agent
    "agent_tagged": list, #List of agents 0 not tagged 1 tagged
    #Teamates First where n is the teammate ID
    "teammate_n_bearing": float, #Agents yaw towards teammate
    "teammate_n_distance": float, #Agents distance towards teammate
    "teammate_n_relative_heading": float, #Teammates current yaw value
    "teammate_n_speed": float, #Teammates current speed
    "teammate_n_has_flag": bool, # True if teammate currently has opponents flag
    "teammate_n_on_side": bool, # True if teammate is on teams side
    "teammate_n_tagging_cooldown": float, #Current value for tagging cooldown
    #Opponents
    "opponent_n_bearing": float, #Agents yaw towards opponent
    "opponent_n_distance": float, #Agents distance towards opponent
    "opponent_n_relative_heading": float, #Opponent current yaw value
    "opponent_n_speed": float, #Opponent current speed
    "opponent_n_has_flag": bool, # True if opponent currently has opponents flag
    "opponent_n_on_side": bool, # True if opponent is on their side
    "opponent_n_tagging_cooldown": float, #Current value for tagging cooldown.

}

#prev_params is the parameters from the previous step and have the same keys as params
"""

import math

def sparse(self, params, prev_params):
    reward = 0
    # Penalize player for opponent grabbing team flag
    if params["opponent_flag_pickup"] and not prev_params["opponent_flag_pickup"]:
        reward += 50
    # Penalize player for opponent successfully capturing team flag
    if params["opponent_flag_capture"] and not prev_params["opponent_flag_capture"]:
        reward +=  100
    # Reward player for grabbing opponents flag
    if params["team_flag_pickup"] and not prev_params["team_flag_pickup"]:
        reward += -50
    # Reward player for capturing opponents flag
    if params["team_flag_capture"] and not prev_params["team_flag_capture"]:
        reward += -100
    # Check to see if agent was tagged
    if params["agent_tagged"][params["agent_id"]]:
        if prev_params["has_flag"]:
            reward += -100
        else:
            reward += -50
    # Check to see if agent tagged an opponent
    tagged_opponent = params["agent_captures"][params["agent_id"]]
    if tagged_opponent is not None:
        if prev_params["opponent_" + str(tagged_opponent) + "_has_flag"]:
            reward += 50
        else:
            reward += 100
    # Penalize agent if it went out of bounds (Hit border wall)
    if params["agent_oob"][params["agent_id"]] == 1:
        reward -= 100
    return reward


def custom_v1(self, params, prev_params):
    return 0

collision_rad = 6.5
def west_point_attack1(self, params, prev_params):
    reward = 0
    # Reward player for grabbing opponents flag
    if params["team_flag_pickup"] and not prev_params["team_flag_pickup"]:
        reward +=  10

    # Reward player for capturing opponents flag
    if params["team_flag_capture"] and not prev_params["team_flag_capture"]:
        reward +=  50
    
    # Penalize if agent was tagged
    if params["agent_tagged"][params["agent_id"]]:
        return -(params["team_flag_home"]/50)**2

    # Reward for goint to the flag
    if not params["team_flag_pickup"] and params["speed"] > 0:
        reward += math.cos(math.radians(params["opponent_flag_bearing"]))/4
        # Closer to the opponent flags 
        if prev_params["opponent_flag_distance"] > params["opponent_flag_distance"]:
            reward += 15/params["opponent_flag_distance"]

    # Reward the flag return home 
    if params["team_flag_pickup"]:
        reward += math.cos(math.radians(params["team_flag_bearing"]))
        if prev_params["team_flag_distance"] > params["team_flag_distance"] and params["speed"] > 0:
            reward += 20/params["team_flag_distance"]
        


    # Avoid the other boat!! (change 1 to the id of the other)
    if params["opponent_0_distance"] < 10.25:
        reward += -2

    # Avoid Collisions!!
    if params["opponent_0_distance"] < collision_rad:
        reward += -1

    # Penalize agent if it went out of bounds (Hit border wall)
    if params["agent_oob"][params["agent_id"]] == 1:
        reward += -1
        if params["opponent_flag_pickup"]:
            reward += -1
    
   # Check to see if agent tagged an opponent (in case the defender is extra aggressive)
    tagged_opponent = params["agent_captures"][params["agent_id"]]
    if tagged_opponent is not None:
        reward += 0.3

    #Penalize for too close to wall
    for key, item in params["wall_distances"].items():
        if(item <= 5):
            reward += -2
            # print("TOO CLOSE")
            break
    # print(reward)
    return reward

def west_point_defend0(self, params, prev_params):
    reward = 0
    # Penalize player for opponent grabbing team flag
    # print("team pick up", params["team_flag_pickup"])
    # print("team capture", params["team_flag_capture"])
    # print("opponent pick up", params["opponent_flag_pickup"])
    # print("opponent capture", params["opponent_flag_capture"])
    # print("opponent_n_has_flag", params["opponent_1_has_flag"])
    if params["opponent_flag_pickup"]:
        reward += -5
    # Penalize player for opponent successfully capturing team flag
    if params["opponent_flag_capture"] and not prev_params["opponent_flag_capture"]:
        reward +=  -50

    # Check to see if agent tagged an opponent
    tagged_opponent = params["agent_captures"][params["agent_id"]]
    if tagged_opponent is not None:
        reward += 2
    
    #In case goes on other side and gets tagged
    if params["agent_tagged"][params["agent_id"]]:
        return -(params["team_flag_home"]/50)**2

    # Reward when facing the opponent
    if params["on_own_side"]:
        reward += math.cos(math.radians(params["opponent_1_bearing"]))/2.5

    #Reward when moving towards the opponent
    
    if prev_params["opponent_1_distance"] > params["opponent_1_distance"] and params["speed"] > 0:
        
        # If on defending side and if they don't have flag
        if not params["opponent_1_on_side"]:
            reward += 10/params["opponent_1_distance"]

    # Avoid Collisions!!
    if params["opponent_1_distance"] < collision_rad:
        reward += -1
    
    #Penalize if not on own side
    if not params["on_own_side"]:
        reward += -1

    # Penalize agent if it went out of bounds (Hit border wall)
    # Only works occasionally for some reason
    if prev_params["agent_oob"][params["agent_id"]] == 1 or params["agent_oob"][params["agent_id"]] == 1:
        reward += -1
        # print("here")
    # print(params["wall_distances"])
    for key, item in params["wall_distances"].items():
        if(item <= 5):
            reward += -1
            # print("TOO CLOSE")
            break
    # print(reward)
    return reward



#print reward when driving with arrow keys
#add no op
#add base policy general
#action based rewards are ok! use dot product to ensure going towards the flag
def wp_attack_multi_0(self, params, prev_params):
    reward = 0
    # Reward player for grabbing opponents flag
    if params["team_flag_pickup"] and not prev_params["team_flag_pickup"]:
        reward +=  10

    # Reward player for capturing opponents flag
    if params["team_flag_capture"] and not prev_params["team_flag_capture"]:
        reward +=  50
    
    # Penalize if agent was tagged
    if params["agent_tagged"][params["agent_id"]]:
        return -(params["team_flag_home"]/50)**2
        

    # Reward the flag return home
    if params["team_flag_pickup"]:
        reward += math.cos(math.radians(params["team_flag_bearing"]))
    if prev_params["team_flag_distance"] > params["team_flag_distance"] and params["thrust"] > 0:
        reward += 20/params["team_flag_distance"] 

    # Reward going towards the flag
    if not params["team_flag_pickup"] and params["thrust"] > 0:
        reward += math.cos(math.radians(params["opponent_flag_bearing"]))/4
        # Closer to the opponent flags 
        if prev_params["opponent_flag_distance"] > params["opponent_flag_distance"]:
            reward += 15/params["opponent_flag_distance"]

    # Avoid the other enemy boats!! 
    if params["opponent_2_distance"] < 10.5 or params["opponent_3_distance"] < 10.5:
        reward += -1

    # Avoid Collisions!!
    if params["opponent_2_distance"] < collision_rad or params["opponent_3_distance"] < collision_rad or params["teammate_1_distance"] < collision_rad:
        reward += -10

    # In case captures on own side
    tagged_opponent = params["agent_captures"][params["agent_id"]]
    if tagged_opponent is not None:
        reward += 2

    # Penalize agent if it went out of bounds (Hit border wall)
    if params["agent_oob"][params["agent_id"]] == 1:
        reward += -1
        if params["team_flag_pickup"]:
            reward += -1

    for key, item in params["wall_distances"].items():
        if(item <= 5):
            reward += -2
    
    return reward

def wp_defend_multi_1(self, params, prev_params):
    reward = 0
    # Penalize player for opponent grabbing team flag
    if params["opponent_flag_pickup"]:
        reward += -5
    # Penalize player for opponent successfully capturing team flag
    if params["opponent_flag_capture"] and not prev_params["opponent_flag_capture"]:
        reward +=  -50

    # Check to see if agent tagged an opponent
    tagged_opponent = params["agent_captures"][params["agent_id"]]
    if tagged_opponent is not None:
            reward += 2

    #In case goes on other side and gets tagged
    if params["agent_tagged"][params["agent_id"]]:
        return -(params["team_flag_home"]/50)**2

    #Rewad facing the closer opponent
    if params["opponent_2_distance"] > params["opponent_3_distance"]:
        reward += math.cos(math.radians(params["opponent_3_bearing"]))/2
        #Reward if closer than previous iteration and they're not in cooldown
        if prev_params["opponent_3_distance"] > params["opponent_3_distance"] and (params["thrust"] > 0 and params["speed"] > 0):
            
            # If on defending side and if they don't have flag
            if params["opponent_3_on_side"] and not params["opponent_3_has_flag"]:
                reward += 10/params["opponent_3_distance"]

    else:
        reward += math.cos(math.radians(params["opponent_2_bearing"]))/2
        #Reward if closer than previous iteration and they're not in cooldown
        if prev_params["opponent_2_distance"] > params["opponent_2_distance"] and (params["thrust"] > 0 and params["speed"] > 0):
            
            # If on defending side and if they don't have flag
            if params["opponent_2_on_side"] and not params["opponent_2_has_flag"]:
                reward += 10/params["opponent_2_distance"]


    # Avoid Collisions!!
    if params["opponent_2_distance"] < collision_rad or params["opponent_3_distance"] < collision_rad or params["teammate_0_distance"] < collision_rad:
        reward += -1

    #Penalize if not on own side
    if not params["on_own_side"]:
        return -1

    # Penalize agent if it went out of bounds (Hit border wall)
    if params["agent_oob"][params["agent_id"]] == 1:
        reward += -1
    
    for key, item in params["wall_distances"].items():
        if(item <= 5):
            reward += -1

    return reward
