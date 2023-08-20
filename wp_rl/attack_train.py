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
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
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
import pyquaticus.base_policies.base_policies as bp 
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
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

    #first half blue (typically defend) and second half red (typically attack)
    reward_config = {0:rew.west_point_defend0, 1:rew.west_point_attack1} # Example Reward Config
    args = parser.parse_args()

    RENDER_MODE = 'None' if args.render else None #set to 'human' if you want rendered output
    
    env_creator = lambda config: pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, reward_config=reward_config, team_size=1)
    env = ParallelPettingZooEnv(pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, reward_config=reward_config, team_size=1))
    register_env('pyquaticus', lambda config: ParallelPettingZooEnv(env_creator(config)))
    
    obs_space = env.observation_space
    act_space = env.action_space

    
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == 0:
            selection = np.random.choice(["noop-policy", "rand-policy", "easy-defend", "medium-defend", "hard-defend"], p=[0.15, 0.2, 0.3, 0.2, 0.15])
            # print(selection)
            return selection #np.random.choice(["attacker-0-policy", "rand-policy"])
        else:
            return "attacker-1-policy"
    
    env.close()
    
    # attack_policy = Policy.from_checkpoint("/home/pmp2149/Documents/pyquaticus_wp/wp_rl/ray_single_attack/checkpoint_010604/policies/attacker-1-policy")

    policies = {
        'noop-policy':(bp.NoOp, obs_space, act_space, {}),
        'attacker-1-policy':(None, obs_space, act_space, {}), 
        'defender-0-policy':(None, obs_space, act_space, {}),
        'rand-policy':(bp.RandPolicy, obs_space, act_space, {}),
        'easy-defend':(bp.DefendGen(1, Team.BLUE_TEAM, 'easy', 1), obs_space, act_space, {}),
        'medium-defend':(bp.DefendGen(1, Team.BLUE_TEAM, 'medium', 1), obs_space, act_space, {}),
        'hard-defend':(bp.DefendGen(1, Team.BLUE_TEAM, 'hard', 1), obs_space, act_space, {})
    }
    # else:
    #     policies = {
    #     'attacker-1-policy':(None, obs_space, act_space, {}), 
    #     'defender-0-policy':(Policy.from_checkpoint('./ray_single_defend/'+str(args.checkpoint)+'/policies/defender-0-policy'), obs_space, act_space, {}),
    #     'rand-policy':(BasePolicyGen.RandPolicy, obs_space, act_space, {}),
    #     'easy-attack':(BasePolicyGen.AttackGen(1, Team.RED_TEAM, 'easy', 1), obs_space, act_space, {}),
    #     'medium-attack':(BasePolicyGen.AttackGen(1, Team.RED_TEAM, 'medium', 1), obs_space, act_space, {}),
    #     'hard-attack':(BasePolicyGen.AttackGen(1, Team.RED_TEAM, 'hard', 1), obs_space, act_space, {})
    #     }
    ppo_config = PPOConfig().environment(env='pyquaticus').rollouts(num_rollout_workers=5).resources(num_cpus_per_worker=1, num_gpus=0)

    # ppo_config = PPOConfig().environment(env='pyquaticus').rollouts(num_rollout_workers=8).resources( num_learner_workers=1, num_gpus_per_learner_worker = 1, num_cpus_per_worker=2, num_gpus=1, num_gpus_per_worker= 0.1)
    ppo_config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=["attacker-1-policy"],)
    algo = ppo_config.build()

    if args.checkpoint != "":
        algo.restore('./ray_single_attack/'+str(args.checkpoint)+'/')
        # algo = Algorithm.from_checkpoint(
        #     checkpoint='./ray_single_attack/'+str(args.checkpoint),
        #     policy_ids={"attacker-1-policy", "noop-policy", "rand-policy", "easy-defend", "medium-defend", "hard-defend"},  # <- restore only those policy IDs here.
        #     policy_mapping_fn=policy_mapping_fn,  # <- use this new mapping fn.
        # )

    for i in range(10000):
        algo.train()
        print("Iter: " + str(i))
        if np.mod(i, 50) == 0 and i>50:
            chkpt_file = algo.save('./ray_single_attack/')
            print(f'Saved to {chkpt_file}', flush=True)

    
