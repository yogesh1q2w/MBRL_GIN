

import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2,PPO1

INorNN = 'NN'
seed = 1
run = 1
total_timesteps = 10000
environment = 'Swimmer-v2'
path = 'Results/'+environment+ '_seed='+str(seed)+'_run='+str(run)+'_total_timesteps='+str(total_timesteps)+ '_ppo1_episode_reward.npy'
pathmodel = 'Results/'+environment+ '_seed='+str(seed)+'_run='+str(run)+'_total_timesteps='+str(total_timesteps)+ '_ppo1'

env = DummyVecEnv([lambda: gym.make(environment)])
# Automatically normalize the input features
env = VecNormalize(env, norm_obs=True, norm_reward=False,
                   clip_obs=10.)

model = PPO1(MlpPolicy, env,verbose  = 0)
model.learn(total_timesteps=total_timesteps,path = path,seed = seed)


# Don't forget to save the running average when saving the agent
log_dir = "/tmp/"
model.save(pathmodel)
env.save_running_average(log_dir)

