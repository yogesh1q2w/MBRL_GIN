

import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

INorNN = 'NN'
seed = 1
run = 1
total_timesteps = 5000000
environment = 'Swimmer-v2'
path = 'Results/'+environment+ '_seed='+str(seed)+'_run='+str(run)+'_total_timesteps='+str(total_timesteps)+ '_ddpg_episode_reward.npy'
pathmodel = 'Results/'+environment+ '_seed='+str(seed)+'_run='+str(run)+'_total_timesteps='+str(total_timesteps)+ '_ddpg'


env = gym.make(environment)
env = DummyVecEnv([lambda: env])

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
n = model.learn_env(total_timesteps=total_timesteps,seed = seed,environment = environment)
model.save(pathmodel)
# np.save("swimmer_ddpg_reward.npy",n)

del model # remove to demonstrate saving and loading
print("Done...")
model = DDPG.load(pathmodel)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
