from Model import Model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym
import random
from gym import utils
from gym.envs.mujoco import mujoco_env

from collections import deque

from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

environment = 'Swimmer-v2'
run = 1
seed = 1
env_timesteps = 1000
model_timesteps  = 1000
path = 'Results/'+environment+ '_seed='+str(seed)+'_run='+str(run)+ '_ddpg_episode_reward.npy'
pathmodel = 'Results/'+environment+ '_seed='+str(seed)+'_run='+str(run)+ '_ddpg'

env = gym.make(environment)
env = DummyVecEnv([lambda: env])

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = Model(environment = environment, list_dense_nodes=[1024,1024])

print('Model initialized')

policy = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
total_steps = 1000

print('DDPG initiliazed')

sample_buffer1 = deque(maxlen = 50000)
sample_buffer2 = deque(maxlen = 50000)
sample_buffer3 = deque(maxlen = 50000)

for _ in range(total_steps):

    print ('Iteration :',  _ + 1)
    print('Training policy on environment')
    samples = policy.learn_env(total_timesteps=env_timesteps,seed = seed,environment = environment,log_interval=10)
    
    env_timesteps = 1000            #change env_steps from next iteration onwards
    
    for z in samples[0]:
        sample_buffer1.append(z)
    for z in samples[1]:
        sample_buffer2.append(z)
    for z in samples[2]:
        sample_buffer3.append(z)
    
    loss = model.train_network(np.array(sample_buffer1),np.array(sample_buffer2),np.array(sample_buffer3),algorithm_id='DDPG', mini_batch_num=1000)


    print('Model train loss = ', loss)

    print('Training policy on model')
    
    policy.learn_model(model,total_timesteps=model_timesteps,seed = seed,path =path,environment = environment,log_interval=10)
    policy.save(pathmodel)

