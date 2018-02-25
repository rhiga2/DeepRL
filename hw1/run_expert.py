#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import roboschool

def run_policy(policy, env, max_timesteps=None, num_rollouts=20, render=False):
    max_steps = max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy.get_action(obs[np.newaxis, :])
            action = action[0]
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    data = {'observations': np.array(observations),
            'actions': np.array(actions)}
    return data

def behavioral_cloning(expert_policy, agent_policy, env, num_epochs=10,
                       max_timesteps=None, num_rollouts=20):
    expert_data = run_policy(expert_policy, env, max_timesteps=max_timesteps,
                             num_rollouts=num_rollouts)
    loss = agent_policy.train_policy(expert_data['observations'], expert_data['actions'], num_epochs)
    return loss

def dagger(expert_policy, agent_policy, env, num_iterations=10,
           num_epochs=4, max_timesteps=None, num_rollouts=20):
    # Generate expert data
    expert_data = run_policy(expert_policy, env,
                             max_timesteps=max_timesteps,
                             num_rollouts=num_rollouts)

    for i in range(num_iterations):
        # Train agent on expert data
        agent_policy.train_policy(expert_data['observations'], expert_data['actions'], num_epochs=num_epochs)

        # Run agent to generate observations
        agent_data = run_policy(agent_policy, env, max_timesteps=max_timesteps, num_rollouts=num_rollouts)

        # Label observations
        expert_actions = expert_policy.get_action(agent_data['observations'])

        # Aggregate observations
        expert_data['observations'] = np.concatenate(
                                          [expert_data['observations'], agent_data['observations']],
                                          axis=0
                                      )
        expert_data['actions'] = np.concatenate(
                                     [expert_data['actions'], expert_actions],
                                     axis=0
                                 )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--algorithm', type=str, default='ep',
                        help='Specifies which algorithm to run')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_epochs', type=int, default=4,
                        help='Number of epochs to run for algorithm')
    args = parser.parse_args()

    env = gym.make(args.envname)

    with tf.Session():
        print('loading and building expert policy')
        expert_policy = load_policy.make_policy(args.expert_policy_file, env)
        print('loading and building expert policy')

        # rollout expert policy
        if args.algorithm == 'ep':
            run_policy(expert_policy, env,
                max_timesteps=args.max_timesteps,
                num_rollouts=args.num_rollouts,
                render=args.render
            )
        # behavioral cloning
        elif args.algorithm == 'bc':
            agent_policy = load_policy.get_new_policy(env, args.lr)

            loss = behavioral_cloning(
                       expert_policy, agent_policy, env,
                       max_timesteps=args.max_timesteps,
                       num_rollouts=args.num_rollouts,
                       num_epochs=args.num_epochs
                   )
            # test agent
            run_policy(agent_policy, env,
                max_timesteps=args.max_timesteps,
                num_rollouts=1,
                render=args.render
            )

        elif args.algorithm == 'dg':
            agent_policy = load_policy.get_new_policy(env, args.lr)

            dagger(
                expert_policy, agent_policy, env,
                num_epochs=args.num_epochs,
                max_timesteps=args.max_timesteps,
                num_rollouts=args.num_rollouts
            )

            # test agent
            run_policy(agent_policy, env,
                max_timesteps=args.max_timesteps,
                num_rollouts=1,
                render=args.render
            )

if __name__ == '__main__':
    main()
