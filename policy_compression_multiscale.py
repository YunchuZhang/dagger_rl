import argparse

import tensorflow as tf 
import load_ddpg
import tqdm
from utils import change_env_to_use_correct_mesh, change_env_to_rescale_mesh
from rollouts import rollout, append_paths
import numpy as np 
from multiprocessing.pool import ThreadPool, Pool

##### Imports related to environment #############
import gym

# push task
meshes = ['keyboard']# 'car2'] #, 'mug1', 'keyboard'] #'car2', 'eyeglass','headphones', 'knife2', 'mug1', 'hamet', 'keyboard']
scales = np.arange(0.3, 2, 0.1)
z_rotation_angles = np.arange(0, 2*np.pi, np.pi/6)

#, 'mouse'] : checkpoint does not load
# 'book' Shapes are [256,2] and [256,3]. for 'Assign_772' (op: 'Assign') with input shapes: [256,2], [256,3].

#pick task
#meshes = ['car2', 'headphones', 'mouse']

#reach task


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--expert_data_path',
						default=None,
						type=str,
						help='Path to some initial expert data collected.')
	parser.add_argument('--expert_log_path',
						default=None,
						type=str,
						help='Path to some initial expert data collected.')
	parser.add_argument('--max-path-length', '-l', type=int, default=50)
	parser.add_argument('--num-rollouts', '-n', type=int, default=500)
	args = parser.parse_args()

	return args

def load_policy(expert_data_path, mesh):
	# Load the policy corresponding to the source mesh
	load_path='{}/{}'.format(args.expert_data_path)+'/save55'
	params_path='{}/{}'.format(args.expert_log_path)

	expert_policy = load_ddpg.load_policy(load_path, params_path)
	return expert_policy

def do_rollouts(env, expert_policy, tid, per_thread_rollouts=5):
	print("This is thread ", tid)
	#change_env_to_use_correct_mesh(mesh)
	#initialize environments with this mesh
	#env = gym.make('SawyerPushAndReachEnvEasy-v0')
	# expert_policy = load_policy(expert_data_path, mesh)
	print("Starting rollouts")
	_, stats = rollout(env,
		per_thread_rollouts,
		args.max_path_length,
		expert_policy, 
		image_env=False)

	env.close()
	#tf.get_variable_scope().reuse_variables()
	return stats['success_rate']

class PolicyCompressor:
	def __init__(self, expert_data_path,
						num_rollouts=100,
						max_path_length=50,
						accept_threshold=0.9,
						num_threads=5):

		self.expert_data_path = expert_data_path
		self.num_rollouts = num_rollouts
		self.max_path_length = max_path_length
		self.policy_bank = {}
		self.clusters = {}
		self.cluster_lookup = {}
		self.clusters_scaled = {}
		self.num_clusters = 0
		self.accept_threshold = accept_threshold
		self.num_threads = num_threads
		self.policy_success_rates = {}

		self.all_accuracies = {}

	def compress_policies_online(self, mesh, scale=1.0):
		# print('Starting rollouts for {} scale {} with angle {}'.format(mesh,"1 1 1",str(0)))
		# # initialize environments with this mesh
		# change_env_to_use_correct_mesh(mesh)
		# change_env_to_rescale_mesh(mesh, scale=1.0)

		# env = gym.make('SawyerPushAndReachEnvEasy-v0', z_rotation_angle=0)
		
		# expert_policy = load_policy(self.expert_data_path, mesh)
		# # perform rollouts to gather stats
		# _, stats = rollout(env,
		# 			args.num_rollouts,
		# 			args.max_path_length,
		# 			expert_policy,
		# 			mesh = mesh,
		# 			image_env=False,
		# 			scale = 1.0)
		# base_success_rate = stats['success_rate']
		# assert base_success_rate == 1.0
		# self.all_accuracies.append((mesh, 1, 0, base_success_rate))

		# success_rates = []
		# mesh_vals = []
		
		change_env_to_use_correct_mesh(mesh)
		change_env_to_rescale_mesh(mesh, scale)

		success_rates = []
		for rot_angle in z_rotation_angles:
			print('Starting rollouts for {} scale {} with angle {}'.format(mesh, scale, rot_angle))
			#initialize environments with this mesh
			env = gym.make('FetchPickAndPlace-v1', z_rotation_angle=rot_angle)

			# load policy of the first mesh (parent mesh) in an existing cluster
			expert_policy = load_policy(self.expert_data_path, mesh)

			# perform rollouts to gather stats
			_, stats = rollout(env,
					args.num_rollouts,
					args.max_path_length,
					expert_policy,
					mesh = mesh, 
					image_env=False,
					scale = scale
					)

			success_rate = stats['success_rate']
			success_rates.append(success_rate)
			self.all_accuracies["{} {} {}".format(mesh, scale, rot_angle)] = success_rate

		best_sr = np.max(success_rates)
		best_rotation = z_rotation_angles[np.argmax(success_rates)]
		self.policy_success_rates["{} {}".format(mesh, scale)] = (best_sr, best_rotation)
		print("Max Success Rate ", best_sr)

	def compress_policies_diff_scales(self):
		for mesh in meshes:
			for n in scales:
				self.compress_policies_online(mesh, scale=n)
			import json
			f1 = "all_accuracies_rinit_scaled{}.json".format(mesh)
			f2 = "best_accuracies_rinit_scaled{}.json".format(mesh)
			with open(f1, 'w') as outfile:
				json.dump(self.all_accuracies, outfile)
			outfile.close()

			with open(f2, 'w') as outfile:
				json.dump(self.policy_success_rates, outfile)
			outfile.close()

		print(self.policy_success_rates)

def main(args):
	compressor = PolicyCompressor(args.expert_data_path, num_rollouts=args.num_rollouts)
	compressor.compress_policies_diff_scales()
	# rescale everything back to 1
	for mesh in meshes:
		change_env_to_rescale_mesh(mesh, 1)

if __name__=="__main__":
	args = parse_args()
	main(args)
