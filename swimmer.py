import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env

env = gym.make('Swimmer-v2')
def data(str):
	f = False
	if str == "train":
		I = 100
		J = 1000
	else:
		I = 50
		J = 2000
	for i in range(I):
		s = env.reset()
		objectrep_s1 = np.array(s)
		for j in range(J):
			# env.render()
			a = env.action_space.sample()
			a = np.array(a)
			s1,r,d,_ = env.step(a) # take a random action
			objectrep_s1  = np.append(objectrep_s1,a)
			objectrep_s2 = np.array(s1)
			if not f:
				input_s_a = np.array([np.array(objectrep_s1)])
				output_s = np.array([np.array(objectrep_s2)])
			else:
				input_s_a = np.append(input_s_a,np.array([objectrep_s1]),axis = 0)
				output_s = np.append(output_s,np.array([objectrep_s2]),axis = 0)
			# print(final,finalAc)
			f = True
			objectrep_s1 = objectrep_s2
			s = s1

	np.save('Data/'+str+'_input_s_a.npy',input_s_a)
	np.save('Data/'+str+'_output_s.npy',output_s)

	env.close()

data('train')
data('test')
data('val')
