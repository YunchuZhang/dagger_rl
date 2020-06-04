import argparse

import tensorflow as tf 
import load_ddpg
import tqdm
from utils import change_env_to_use_correct_mesh
from rollouts import rollout, append_paths
import numpy as np 
from multiprocessing.pool import ThreadPool, Pool

##### Imports related to environment #############
import gym
import multiworld

multiworld.register_all_envs()

meshes = ['car2', 'eyeglass','headphones', 'knife2', 'mug1'] #'mouse'

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

def load_policy(expert_data_path, mesh):
	# Load the policy corresponding to the source mesh
	load_path='{}/{}'.format(args.expert_data_path, mesh)+'/save150'
	params_path='{}/{}'.format(args.expert_data_path, mesh)

	expert_policy = load_ddpg.load_policy(load_path, params_path)
	return expert_policy

def change_orientation():
	# how to make the change in orientation reflect back on 
	pass

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
		self.num_clusters = 0
		self.accept_threshold = accept_threshold
		self.num_threads = num_threads
		# Access to a minimal policy bank, also has information about which meshes to run on which policy
		# Takes in a new object and determines if it can be merged with an existing policy or can spawn a new policy
		# input: new object that needs to be classified

	def compress_policies_online(self, mesh):
		per_thread_rollouts = self.num_rollouts/self.num_threads

		if len(self.policy_bank.keys()) == 0:
			self.policy_bank[mesh] = load_policy(self.expert_data_path, mesh)
			self.num_clusters += 1
			self.clusters["c" + str(self.num_clusters)] = [mesh]
			self.cluster_lookup[mesh] = "c" + str(self.num_clusters)
		else:
			success_rates = []
			mesh_vals = []

			print("Starting rollouts for src {} tgt {} ".format(mesh, mesh))
			
			change_env_to_use_correct_mesh(mesh)
			#initialize environments with this mesh
			env = gym.make('SawyerPushAndReachEnvEasy-v0')
			
			expert_policy = load_policy(self.expert_data_path, mesh)
			# # perform rollouts to gather stats
			# _, stats = rollout(env,
			# 			args.num_rollouts,
			# 			args.max_path_length,
			# 			expert_policy,
			# 			mesh = mesh, 
			# 			image_env=False)
			# base_success_rate = stats['success_rate']

			pool = ThreadPool(processes=self.num_threads)
			async_results = [pool.apply_async(do_rollouts, (env, expert_policy, i, per_thread_rollouts,)) for i in range(self.num_threads)]

			
			total_success_rate = 0
			for i in range(self.num_threads):
				total_success_rate += async_results[i].get()
			base_success_rate = total_success_rate/self.num_threads
			pool.join()
			
			# Compute accuracy for each of the clusters
			for cid, meshes in self.clusters.items():
				# load policy of the first mesh (parent mesh) in an existing cluster
				expert_policy = load_policy(self.expert_data_path, meshes[0])
				print("Checking performance of {} on policy for {}".format(mesh, meshes[0]))
				# # perform rollouts to gather stats
				# _, stats = rollout(env,
				# 		args.num_rollouts,
				# 		args.max_path_length,
				# 		expert_policy,
				# 		mesh = meshes[0], 
				# 		image_env=False)

				pool = ThreadPool(processes=self.num_threads)
				async_results = [pool.apply_async(do_rollouts, (env, expert_policy, i, per_thread_rollouts,)) for i in range(self.num_threads)]

				total_success_rate = 0
				for i in range(self.num_threads):
					total_success_rate += async_results[i].get()
				success_rate = total_success_rate/self.num_threads

				pool.join()
				mesh_vals.append(meshes[0])
				success_rates.append(success_rate)

			max_success_rate = np.max(success_rates)

			print("Max Success Rate ", max_success_rate, " Base success rate ", base_success_rate)
			if max_success_rate >= self.accept_threshold: #* base_success_rate:
				max_idx = np.argmax(success_rates)
				best_mesh = mesh_vals[max_idx]
				cval = self.cluster_lookup[best_mesh]
				self.clusters[cval].append(mesh)
			else:
				# Form a new cluster
				self.policy_bank[mesh] = load_policy(self.expert_data_path, mesh)
				self.num_clusters += 1
				self.clusters["c" + str(self.num_clusters)] = [mesh]
				self.cluster_lookup[mesh] = "c" + str(self.num_clusters)


	def compress_policies(self):
		for mesh in meshes:
			self.compress_policies_online(mesh)
		print("Compressed Meshes")
		print(self.clusters)
		print("Percentage compression for threshold {} : {} ".format(self.accept_threshold, self.num_clusters/len(meshes)))

def main(args):
	compressor = PolicyCompressor(args.expert_data_path, num_rollouts=args.num_rollouts)
	compressor.compress_policies()

if __name__=="__main__":
	args = parse_args()
	main(args)
