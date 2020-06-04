import argparse

import tensorflow as tf 
import load_ddpg
import tqdm
from utils import change_env_to_use_correct_mesh
from rollouts import rollout, append_paths
import numpy as np 
from multiprocessing.pool import ThreadPool, Pool

import matplotlib
# matplotlib.use('tkagg')

import matplotlib.pyplot as plt

##### Imports related to environment #############
import gym
import multiworld

multiworld.register_all_envs()

# push task
meshes = ['car2','mug1'] #'''eyeglass','headphones', 'knife2', 'mug1', 'hamet', 'keyboard']

z_rotation_angles = np.arange(0, 2*np.pi, np.pi/6)

import os
if not os.path.exists("run_policies"):
    os.makedirs("run_policies")

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--expert_data_path',
						default=None,
						type=str,
						help='Path to some initial expert data collected.')
	parser.add_argument('--max-path-length', '-l', type=int, default=50)
	parser.add_argument('--num-rollouts', '-n', type=int, default=500)
	args = parser.parse_args()

	return args

def load_policy(mesh):
	# Load the policy corresponding to the source mesh
	load_path='{}/{}'.format(args.expert_data_path, mesh)+'/save150'
	params_path='{}/{}'.format(args.expert_data_path, mesh)

	expert_policy = load_ddpg.load_policy(load_path, params_path)
	return expert_policy

# src_mesh: object whose policy will be loaded
# tgt_mesh: object in environment
def execute_policies(src_mesh, tgt_mesh, num_rollouts=100, num_timesteps=50):
	change_env_to_use_correct_mesh(tgt_mesh)

	sr_vals = {}
	for rot_angle in z_rotation_angles:
		print('Starting rollouts for {} on policy for {} with angle {}'.format(tgt_mesh, src_mesh, rot_angle))
		#initialize environments with this mesh
		env = gym.make('SawyerPushAndReachEnvEasy-v0', z_rotation_angle=rot_angle)
		
		expert_policy = load_policy(src_mesh)

		actor = expert_policy.step

		observation = env.reset()
		success_rate = 0
		for r in range(num_rollouts):
			for t in range(num_timesteps):
				action,_,_,_ = actor(observation)
				# import pdb; pdb.set_trace()
				img = env.render(mode='rgb_array')
				plt.imsave("run_policies/src_{}tgt_{}_r{}_t{}.png".format(src_mesh, tgt_mesh, r, t), img)
				# plt.show(block=False)
				# plt.pause(1)
				# plt.close()
				observation, reward, terminal, info = env.step(action)
				if reward == 0:
					success_rate += 1
					break;

		success_rate /= num_rollouts
		sr_vals[rot_angle] = success_rate

		env.close()

	print("Src mesh {} Tgt mesh {}".format(src_mesh, tgt_mesh))
	print(sr_vals)

def main(args):
	execute_policies("car2", "mug1", args.num_rollouts, args.max_path_length)

if __name__=="__main__":
	args = parse_args()
	main(args)


