import numpy as np
import torch
import gymnasium as gym
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
from torch import autograd
from scipy.spatial.distance import squareform, pdist
from functools import partial
import math
from svpg_setup import *

if __name__ == "__main__":
       
       # REINFORCE

       ENV_NAME = "Acrobot-v1"
       N_PARTICLES = 16
       MAX_EPISODES = 100
       MAX_STEPS = 500
       LEARNING_RATE = 1e-3
       N_ROLLOUTS = 5
       GAMMA = 0.99

       N_HIDDEN = 128

       reinforce = REINFORCE(gamma=GAMMA, gym_env_name=ENV_NAME, n_particles = N_PARTICLES,
                            max_episodes=MAX_EPISODES, max_steps=MAX_STEPS,
                            learning_rate=LEARNING_RATE, num_rollouts=N_ROLLOUTS)
       reinforce_rewards = reinforce.train()

       print("="*5, "\tREINFORCE evaluations\t", "="*5)
       EVAL_EPISODES = 100
       agent_rewards = reinforce.evaluate_policies(EVAL_EPISODES, MAX_STEPS)
       reinforce_avg_reward = np.mean(agent_rewards)
       reinforce_best_reward = reinforce.best_policy.evaluate_policy(env_name=ENV_NAME, n_eval_episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
       reinforce_ensemble_reward = reinforce.evaluate_ensemble_policy(eval_episodes=100)
       print(f"REINFORCE rewards:\n\tRewards for each agent: {agent_rewards}\n\tAverage reward:\
              {reinforce_avg_reward}\n\tBest reward: {reinforce_best_reward}\n\tEnsemble reward: {reinforce_ensemble_reward}")


       # Vanilla SVPG

       ENV_NAME = "Acrobot-v1"
       N_PARTICLES = 16
       MAX_EPISODES = 100
       MAX_STEPS = 500
       LEARNING_RATE = 1e-3
       N_ROLLOUTS = 5
       GAMMA = 0.99
       ALPHA = 10
       N_HIDDEN = 128
       DECAY = 1 - 25 / MAX_EPISODES

       svpg = SVPG_REINFORCE(gamma=GAMMA, alpha=ALPHA, n_particles=N_PARTICLES, gym_env_name=ENV_NAME, n_hidden=N_HIDDEN,
                            learning_rate=LEARNING_RATE, max_episodes=MAX_EPISODES, max_steps=MAX_STEPS, decay=DECAY, num_rollouts=N_ROLLOUTS)

       svpg_rewards = svpg.train()

       print("="*5, "\SVPG evaluations\t", "="*5)
       EVAL_EPISODES = 100
       agent_rewards = svpg.evaluate_policies(EVAL_EPISODES, MAX_STEPS)
       svpg_avg_reward = np.mean(agent_rewards)
       svpg_best_reward = svpg.best_policy.evaluate_policy(env_name=ENV_NAME, n_eval_episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
       svpg_ensemble_reward = svpg.evaluate_ensemble_policy(eval_episodes=100)
       print(f"Vanilla SVPG rewards:\n\tRewards for each agent: {agent_rewards}\n\tAverage reward:\
              {svpg_avg_reward}\n\tBest reward: {svpg_best_reward}\n\tEnsemble reward: {svpg_ensemble_reward}")


       # MK-SVPG

       ENV_NAME = "Acrobot-v1"
       N_PARTICLES = 16
       MAX_EPISODES = 100
       MAX_STEPS = 500
       LEARNING_RATE = 1e-3
       N_ROLLOUTS = 5
       GAMMA = 0.99
       ALPHA = 10
       N_HIDDEN = 128
       DECAY = 1 - 25 / MAX_EPISODES

       B_SCALES = [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0]

       mk_svpg = MK_SVPG_REINFORCE(gamma=GAMMA, alpha=ALPHA, n_particles=N_PARTICLES, gym_env_name=ENV_NAME, n_hidden=N_HIDDEN,
                                   learning_rate=LEARNING_RATE, max_episodes=MAX_EPISODES, max_steps=MAX_STEPS,
                                   bandwidth_scales=B_SCALES, num_rollouts=N_ROLLOUTS, decay=DECAY)

       mk_svpg_rewards = mk_svpg.train()

       print("="*5, "\MK-SVPG evaluations\t", "="*5)
       EVAL_EPISODES = 100
       agent_rewards = mk_svpg.evaluate_policies(EVAL_EPISODES, MAX_STEPS)
       mk_svpg_avg_reward = np.mean(agent_rewards)
       mk_svpg_best_reward = mk_svpg.best_policy.evaluate_policy(env_name=ENV_NAME, n_eval_episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
       mk_svpg_ensemble_reward = mk_svpg.evaluate_ensemble_policy(eval_episodes=100)
       print(f"MK-SVPG rewards:\n\tRewards for each agent: {agent_rewards}\n\tAverage reward:\
              {mk_svpg_avg_reward}\n\tBest reward: {mk_svpg_best_reward}\n\tEnsemble reward: {mk_svpg_ensemble_reward}")


       # Learning curves

       plt.plot(reinforce_rewards, label="REINFORCE")
       plt.plot(svpg_rewards, label="SVPG")
       plt.plot(mk_svpg_rewards, label="MK-SVPG")
       plt.legend()
       plt.title("Acrobot results")
       plt.xlabel("Episodes")
       plt.ylabel("Average reward across agents")
       plt.savefig("./figures/acrobot.png")
       plt.show()