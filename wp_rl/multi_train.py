# (C) 2021 Massachusetts Institute of Technology.

# Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

# The software/firmware is provided to you on an As-Is basis

# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause
import argparse
import gymnasium as gym
import numpy as np
import pygame
from pygame import KEYDOWN, QUIT, K_ESCAPE
import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPOTF1Policy, PPOTorchPolicy
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from ray.rllib.algorithms.algorithm import Algorithm
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOTF2Policy, PPOConfig
from ray.rllib.policy.policy import PolicySpec
import os
import pyquaticus.utils.rewards as rew
from ray.rllib.policy.policy import PolicySpec, Policy
import pyquaticus.base_policies.base_policies as bp






if __name__ == '__main__':
    #ray.init(ignore_reinit_error=True, num_cpus=4)
    
    parser = argparse.ArgumentParser(description='Train a 1v1 policy in a 1v1 PyQuaticus environment')
    parser.add_argument('--render', help='Enable rendering', action='store_true')


    parser.add_argument("--num_gpus", type=int, default=2)

    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Restore previous checkpoint.",
    )

    #first half blue and second half red
    reward_config = {
        0:rew.wp_attack_multi_0, 
        1:rew.wp_defend_multi_1, 
        2:None, 
        3:None
    } 
    args = parser.parse_args()

    RENDER_MODE = 'None' if args.render else None #set to 'human' if you want rendered output
    
    env_creator = lambda config: pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, reward_config=reward_config, team_size=2)
    env = ParallelPettingZooWrapper(pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, reward_config=reward_config, team_size=2))
    register_env('pyquaticus', lambda config: ParallelPettingZooWrapper(env_creator(config)))
    
    obs_space = env.observation_space
    act_space = env.action_space

            
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # print(agent_id)
        if agent_id == 0:
            return "attacker-multi"
        elif agent_id == 1:
            return "defender-multi"
        elif agent_id == 2:
            return np.random.choice(["rand-policy", "noop-policy", "easy-attack", "medium-attack", "hard-attack", "easy-combined2", "medium-combined2", "hard-combined2", "attacker-multi"])
        else:
            return np.random.choice(["rand-policy", "noop-policy", "easy-defend", "medium-defend", "hard-defend", "easy-combined3", "medium-combined3", "hard-combined3", "defender-multi"])
    env.close()
    policies = {
        'attacker-multi':(None, obs_space, act_space, {}), 
        'defender-multi':(None, obs_space, act_space, {}), 

        'rand-policy':(bp.RandPolicy, obs_space, act_space, {}),
        'noop-policy':(bp.NoOp, obs_space, act_space, {}),

        'easy-defend':(bp.DefendGen(3, Team.RED_TEAM, 'easy', 2), obs_space, act_space, {}),
        'medium-defend':(bp.DefendGen(3, Team.RED_TEAM, 'medium', 2), obs_space, act_space, {}),
        'hard-defend':(bp.DefendGen(3, Team.RED_TEAM, 'hard', 2), obs_space, act_space, {}),

        'easy-attack':(bp.AttackGen(2, Team.RED_TEAM, 'easy', 2), obs_space, act_space, {}),
        'medium-attack':(bp.AttackGen(2, Team.RED_TEAM, 'medium', 2), obs_space, act_space, {}),
        'hard-attack':(bp.AttackGen(2, Team.RED_TEAM, 'hard', 2), obs_space, act_space, {}),

        'easy-combined2':(bp.CombinedGen(2, Team.RED_TEAM, 'easy', 2), obs_space, act_space, {}),
        'medium-combined2':(bp.CombinedGen(2, Team.RED_TEAM, 'medium', 2), obs_space, act_space, {}),
        'hard-combined2':(bp.CombinedGen(2, Team.RED_TEAM, 'hard', 2), obs_space, act_space, {}),

        'easy-combined3':(bp.CombinedGen(3, Team.RED_TEAM, 'easy', 2), obs_space, act_space, {}),
        'medium-combined3':(bp.CombinedGen(3, Team.RED_TEAM, 'medium', 2), obs_space, act_space, {}),
        'hard-combined3':(bp.CombinedGen(3, Team.RED_TEAM, 'hard', 2), obs_space, act_space, {})
    }

    
    #CPU training
    ppo_config = PPOConfig().environment(env='pyquaticus').rollouts(num_rollout_workers=1).resources(num_cpus_per_worker=1, num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    #GPU training
    # ppo_config = PPOConfig().environment(env='pyquaticus').rollouts(num_rollout_workers=5).resources( num_learner_workers=2, num_gpus_per_learner_worker = 1, num_cpus_per_worker=2, num_gpus=0.5, _fake_gpus = True )
    ppo_config.multi_agent(policies=policies, \
                           policy_mapping_fn=policy_mapping_fn, \
                            policies_to_train=["attacker-multi", "defender-multi"],)
    algo = ppo_config.build()

    if args.checkpoint != "":
        algo.restore('./ray_wp_multi/'+str(args.checkpoint)+'/')
        # algo = Algorithm.from_checkpoint(
        #     checkpoint='./ray_wp_multi/'+str(args.checkpoint)+'/',
        #     policy_ids={"attacker-multi", "defender-multi"},  # <- restore only those policy IDs here.
        #     policy_mapping_fn=policy_mapping_fn,  # <- use this new mapping fn.
        # )
    s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"
    i = 0 
    while True:
        result = algo.train()
        print("Iter: " + str(i + 1))
        if np.mod(i, 100) == 0:
            chkpt_file = algo.save('./ray_wp_multi/')
            print(s.format(
                i + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
            ))
