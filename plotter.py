import matplotlib.pyplot as plt 
import numpy as np 
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-f',dest = "path", help="The numpy file path you want to plot")
parser.add_argument('-a', dest = "alg", help = "Algorithm used(ppo,trpo,etc)")
args = parser.parse_args()

print(args.path)

a = np.load(args.path)
print(np.shape(a))

fig, ax = plt.subplots(figsize=(7,3))
ax.plot(a,label='Mean Reward per episode', linewidth=2, color='g')

ax.set_xlabel('Episodes')
ax.set_ylabel('Reward')
ax.set_title(args.alg)

fig.savefig('Results/Reward_'+args.alg +'_.png', dpi=200, bbox_inches='tight')
plt.show()

