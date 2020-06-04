import argparse
import os
import json
import pickle
import numpy as np
import tqdm
from xml.etree import ElementTree as et

import tensorflow as tf
import multiworld
import gym
import load_expert
import tf_utils as tfu

from policies.xyz_xyz_policy import XYZ_XYZ_Policy
from rollouts import rollout, append_paths
from softlearning.environments.gym.wrappers import NormalizeActionWrapper
from utils import change_env_to_use_correct_mesh

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

multiworld.register_all_envs()

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env',
						type=str,
						default='SawyerPushAndReachEnvEasy-v0',
						help='Environment we are trying to run.')
	parser.add_argument('--mesh',
                        default=None,
                        type=str,
                        help="Mesh used for Sawyer Task.")
	parser.add_argument('--checkpoint_path',
    					type=str,
    					help='Path to the checkpoint.')
	parser.add_argument('--goal_type',
    					type=str,
    					default='xyz',
    					choices=['xyz', '3d'])
	parser.add_argument('--obs_type',
    					type=str,
    					default='xyz',
    					choices=['xyz', '3d'])
	parser.add_argument('--expert_data_path',
						default=None,
						type=str,
						help='Path to some initial expert data collected.')
	parser.add_argument('--max-path-length', '-l', type=int, default=100)
	parser.add_argument('--num-rollouts', '-n', type=int, default=10)
	parser.add_argument('--num-iterations', type=int, default=50)
	parser.add_argument('--mb_size', type=int, default=64)

	args = parser.parse_args()

	return args

def main(args):
	expert_list = ['mouse','headphones']
	
	## Define environment
	if args.mesh is not None: change_env_to_use_correct_mesh(args.mesh)

	# Dictionary of values to plot
	plotters = {'min_return': [],
				'max_return': [],
				'mean_return': [],
				'mean_final_success': []}

	## Define expert
	expert_policy, env = load_expert.get_policy(args.checkpoint_path, image_env=False)

	## Define policy network
	policy = XYZ_XYZ_Policy("dagger_xyz_xyz", env)

	## Define DAGGER loss
	ob = tfu.get_placeholder(name="ob",
							dtype=tf.float32,
							shape=[None, policy.obs_dim])
	act = tfu.get_placeholder(name="act",
							dtype=tf.float32,
							shape=[None, policy.act_dim])
	loss = tf.reduce_mean(tf.squared_difference(policy.ac, act))

	# lr 0.002 0.001
	# decay 0.96 0.8

	lr = tf.train.exponential_decay(learning_rate = 0.001,
									global_step = step,
									decay_steps = 20000,
									decay_rate = 0.75,
									staircase=True)

	opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

	# Start session
	session = tfu.make_session(num_cpu=2)
	session.__enter__()
	session.run(tf.global_variables_initializer())

	# Load expert policy
	pickle_path = os.path.join(args.checkpoint_path, 'checkpoint.pkl')
	with open(pickle_path, 'rb') as f:
		picklable = pickle.load(f)

	expert_policy.set_weights(picklable['policy_weights'])
	expert_policy.set_deterministic(True).__enter__()

	# Collect initial data
	if args.expert_data_path is None:
		data, _ = rollout(env,
					args.num_rollouts,
					args.max_path_length,
					expert_policy)
		# np.save('expert_data_{}.npy'.format(args.env), data)
	else:
		data = np.load(args.expert_data_path, allow_pickle=True).item()
		roll, _ = rollout(env,
				args.num_rollouts,
				args.max_path_length,
				expert_policy)
	exit()
	## Start training

	# Start for loop
	for i in tqdm.tqdm(range(args.num_iterations)):
		# print('\nIteration {} :'.format(i+1))
		# Parse dataset for supervised learning
		num_samples = data['state_observation'].shape[0]
		idx = np.arange(num_samples)
		np.random.shuffle(idx)
		for j in range(num_samples // args.mb_size):
			np.random.shuffle(idx)
			obs_train = policy.train_process_observation(data, idx[:args.mb_size])
			act_train = data['actions'][idx[:args.mb_size]]
			session.run(opt, feed_dict={ob:obs_train, act:act_train})
		# Perform rollouts
		roll, plot_data = rollout(env,
				args.num_rollouts,
				args.max_path_length,
				policy,
				expert_policy)
		data = append_paths(data, roll)
		for key in plotters.keys(): plotters[key].append(plot_data[key])

	# Plotting results
	color_list = ["#363737"]
	plt.figure(figsize=(4,4))
	plt.rcParams["axes.edgecolor"] = "0.15"
	plt.rcParams["axes.linewidth"]  = 0.5
	plt.rcParams["font.sans-serif"] = "Helvetica"
	plt.rcParams["font.family"] = "sans-serif"
	plt.rcParams["ytick.labelsize"] = "medium"
	plt.rcParams["xtick.labelsize"] = "medium"
	plt.rcParams["font.size"] = 8.3
	for i, key in enumerate(plotters.keys()):
		ax = plt.subplot(2,2,i+1)
		plt.plot(range(args.num_iterations), plotters[key])
		plt.title(key)
	plt.tight_layout()
	plt.savefig('metrics.png', dpi=300)
	plt.close()

	# tf.get_default_session().close()

if __name__ == '__main__':
	args = parse_args()
	main(args)
