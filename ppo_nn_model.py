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

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2,PPO1

environment = 'Swimmer-v2'
run = 1
seed = 1
env_timesteps = 10000
model_timesteps  = 1000
envt_path = 'Results/'+environment+ '_Envt_seed='+str(seed)+'_run='+str(run)+ '_ppo1_episode_reward.npy'
model_path = 'Results/'+environment+ '_Model_seed='+str(seed)+'_run='+str(run)+ '_ppo1_episode_reward.npy'
pathmodel = 'Results/'+environment+ '_seed='+str(seed)+'_run='+str(run)+ '_ppo1'


env = DummyVecEnv([lambda: gym.make(environment)])
# Automatically normalize the input features
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
model = Model(environment = environment, list_dense_nodes=[1024,1024])

print('Model initialized')

policy = PPO1(MlpPolicy, env,verbose  = 1)
total_steps = 1000

print('PPO1 initialized')

sample_buffer1 = deque(maxlen = 50000)
sample_buffer2 = deque(maxlen = 50000)
sample_buffer3 = deque(maxlen = 50000)

for _ in range(total_steps):

    print ('Iteration :',  _ + 1)
    print('Training policy on environment')
    samples = policy.learn_env(total_timesteps=env_timesteps,path =  envt_path, seed = seed, log_interval=10)

    # print (np.shape(samples[0]),np.shape(samples[1]),np.shape(samples[2]))
    
    env_timesteps = 1000            #change env_steps from next iteration onwards
    
    for z in samples[0]:
        sample_buffer1.append(z)
    for z in samples[1]:
        sample_buffer2.append(z)
    for z in samples[2]:
        sample_buffer3.append(z)
    
    loss = model.train_network(np.array(sample_buffer1),np.array(sample_buffer2),np.array(sample_buffer3),algorithm_id='PPO1', mini_batch_num=1000)


    print('Model train loss = ', loss)

    print('Training policy on model')
    
    policy.learn_model(model,total_timesteps=env_timesteps,path =  model_path, seed = seed, log_interval=10)
    policy.save(pathmodel)

