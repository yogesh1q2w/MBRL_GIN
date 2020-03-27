import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import TRPO


INorNN = 'NN'
seed = 1
run = 1
total_timesteps = 5000000
environment = 'Swimmer-v2'
path = 'Results/'+environment+ '_seed='+str(seed)+'_run='+str(run)+'_total_timesteps='+str(total_timesteps)+ '_trpo_episode_reward.npy'
pathmodel = 'Results/'+environment+ '_seed='+str(seed)+'_run='+str(run)+'_total_timesteps='+str(total_timesteps)+ '_trpo'

env = gym.make(environment)
env = DummyVecEnv([lambda: env])

# Automatically normalize the input features
env = VecNormalize(env, norm_obs=True, norm_reward=False,
                   clip_obs=10.)


model = TRPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=total_timesteps,path = path,seed = seed)
model.save(pathmodel)

# Don't forget to save the running average when saving the agent
log_dir = "/tmp/"
env.save_running_average(log_dir)

'''
del model # remove to demonstrate saving and loading
'''
model = TRPO.load("")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
