import argparse
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import pickle
import numpy as np
import tqdm
# from xml.etree import ElementTree as et

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import multiworld
import gym
import load_ddpg
import policies.tf_utils as tfu

from multiworld.envs.mujoco.cameras import init_multiple_cameras
from policies.xyz_xyz_policy import XYZ_XYZ_Policy
from rollouts import rollout, append_paths
from softlearning.environments.gym.wrappers import NormalizeActionWrapper
from utils import change_env_to_use_correct_mesh

multiworld.register_all_envs()

from multiprocessing.pool import ThreadPool, Pool

## Define environment
expert_list = ['car2', 'eyeglass','headphones']#, 'knife2', 'mouse', 'mug1']

def do_rollouts(mesh, tid, per_thread_rollouts=5, policy=None):
	print("This is thread ", tid)

	## Define expert
	load_path='/projects/katefgroup/sawyer_ddpg_weight/{}'.format(mesh)+'/save150'
	params_path='/projects/katefgroup/sawyer_ddpg_weight/{}'.format(mesh)

	expert_policy = load_ddpg.load_policy(load_path, params_path)

	print("Initializing gym environment")
	env = gym.make("SawyerPushAndReachEnvEasy-v0",reward_type='puck_success')

	print("Starting rollouts")
	if policy is None:
	# Collect initial data
		roll, _ = rollout(env,
			per_thread_rollouts,
			args.max_path_length,
			expert_policy,
			mesh = mesh, 
			image_env=False)
	else:
		print("Performing rollouts after training....")
		roll, _ = rollout(env,
			args.num_dagger_rollouts,
			args.max_path_length,
			policy,
			expert_policy,
			mesh = mesh, 
			image_env=False)

	env.close()
	tf.get_variable_scope().reuse_variables()
	return roll

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
	parser.add_argument('--max-path-length', '-l', type=int, default=50)
	parser.add_argument('--num-rollouts', '-n', type=int, default=500)
	parser.add_argument('--num-dagger-rollouts', '-n-dgr', type=int, default=50)
	parser.add_argument('--test-num-rollouts', '-tn', type=int, default=20)
	parser.add_argument('--num-iterations', type=int, default=50)
	parser.add_argument('--mb_size', type=int, default=8)
	parser.add_argument('--checkpoint_freq', type=int, default=5)
	parser.add_argument('--reward_type', type=str, default='puck_success')
	parser.add_argument('--test-policy', type=bool, default=False)
	args = parser.parse_args()

	return args

def main(args):

	# Dictionary of values to plot
	plotters = {'min_return': [],
				'max_return': [],
				'mean_return': [],
				'mean_final_success': []}


	name = "baseline_3obj_no_staircase_lr_0.005_steps_20000_decay_0.96_shuffle"
	log_dir_ = os.path.join("logs_sawyer_baseline_dagger", name)
	checkpoint_dir_ = os.path.join("ckpt_sawyer_baseline_dagger", name)
	set_writer = tf.summary.FileWriter(log_dir_ + '/train', None)

	## Define expert
	env = gym.make("SawyerPushAndReachEnvEasy-v0",reward_type=args.reward_type)

	## Define policy network
	policy = XYZ_XYZ_Policy("dagger_xyz_xyz", env, hidden_sizes=[64, 64, 32])

	## Define DAGGER loss
	ob = tfu.get_placeholder(name="ob",
							dtype=tf.float32,
							shape=[None, policy.obs_dim])

	## Define DAGGER loss
	act = tfu.get_placeholder(name="act",
							dtype=tf.float32,
							shape=[None, policy.act_dim])

	min_return = tfu.get_placeholder(dtype=tf.float32, shape=None, name="min_return")
	max_return = tfu.get_placeholder(dtype=tf.float32, shape=None, name="max_return")
	mean_return = tfu.get_placeholder(dtype=tf.float32, shape=None, name="mean_return")
	mean_final_success = tfu.get_placeholder(dtype=tf.float32, shape=None, name="mean_final_success")

	step = tf.Variable(0, trainable=False)

	# lr 0.002 0.001
	# decay 0.96 0.8

	lr = tf.train.exponential_decay(learning_rate = 0.001,
									global_step = step,
									decay_steps = 20000,
									decay_rate = 0.75,
									staircase=False)

	# Exclude map3D network from gradient computation
	freeze_patterns = []
	freeze_patterns.append("feat")

	loss = tf.reduce_mean(tf.squared_difference(policy.ac, act))
	train_vars = tf.contrib.framework.filter_variables( tf.trainable_variables(),
														exclude_patterns=freeze_patterns)
	# tf.get_variable_scope().reuse_variables()
	# lr = 1e-3
	opt = tf.train.AdamOptimizer(learning_rate=lr, name="adam_optimizer").minimize(loss,
														var_list=train_vars,
														global_step=step)

	# Start session
	session = tfu.make_session(num_cpu=2)
	session.__enter__()
	
	session = tf.get_default_session()
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	session.run(init_op)

	loss_op = tf.summary.scalar('loss', loss)

	with tf.variable_scope("policy_perf"):
		min_return_op = tf.summary.scalar('min_return', min_return)
		max_return_op = tf.summary.scalar('max_return', max_return)
		mean_return_op = tf.summary.scalar('mean_return', mean_return)
		mean_final_success_op = tf.summary.scalar('mean_final_success', mean_final_success)

	saver = tf.train.Saver()

	# Load expert policy
	data = None
	num_threads = 5
	per_thread_rollouts = args.num_rollouts / num_threads
	for mesh in expert_list:
		print('generating {} data'.format(mesh))
		change_env_to_use_correct_mesh(mesh)

		## Define expert
		# /projects/katefgroup/sawyer_ddpg_weight
		# load_path='{}/{}'.format(args.expert_data_path, mesh)+'/save150'
		# params_path='{}/{}'.format(args.expert_data_path, mesh)

		# expert_policy = load_ddpg.load_policy(load_path, params_path)
		# env = gym.make("SawyerPushAndReachEnvEasy-v0", reward_type=args.reward_type)
		
		# # Collect initial data
		# if init is True:
		# 	data, _ = rollout(env,
		# 				args.num_rollouts,
		# 				args.max_path_length,
		# 				expert_policy,
		# 				mesh = mesh, image_env=False)
		# 	np.save('expert_data_{}.npy'.format(args.env), data)
		# 	init = False
		# else:
		# 	roll, _ = rollout(env,
		# 			args.num_rollouts,
		# 			args.max_path_length,
		# 			expert_policy,
		# 			mesh = mesh, image_env=False)
		# 	data = append_paths(data, roll)
		# env.close()
		# tf.get_variable_scope().reuse_variables()


		pool = ThreadPool(processes=num_threads)
		async_results = [pool.apply_async(do_rollouts, (mesh, i, per_thread_rollouts,)) for i in range(num_threads)]

		all_rollouts = []
		for i in range(num_threads):
			all_rollouts.append(async_results[i].get())

		for roll in all_rollouts:
			data = append_paths(data, roll)

	## Start training

	#import pdb; pdb.set_trace()
	# Start for loop
	global_step = 0

	for i in tqdm.tqdm(range(args.num_iterations)):
		# Parse dataset for supervised learning
		num_samples = data['desired_goal_abs'].shape[0]
		print('num_samples',num_samples)
		idx = np.arange(num_samples)
		np.random.shuffle(idx)
		# Make one whole traversal through the dataset for training
		for j in range(num_samples // args.mb_size):
			# st = i * args.mb_size
			# end = min(num_samples, (i+1) * args.mb_size)
			# obs = policy.train_process_observation(data, idx[st:end])
			# act_train = data['actions'][idx[st:end]]
			np.random.shuffle(idx)
			obs = policy.train_process_observation(data, idx[:args.mb_size])
			act_train = data['actions'][idx[:args.mb_size]]
			loss, _ = session.run([loss_op,opt], feed_dict={ob:obs, act:act_train})
			set_writer.add_summary(loss, global_step=global_step)
			global_step = global_step + 1

		# Generate some new dagger data after every few training iterations
		# Perform rollouts
		for mesh in expert_list:
			print('generating {} dagger data'.format(mesh))
			change_env_to_use_correct_mesh(mesh)
			# ## Define expert
			# load_path='{}/{}'.format(args.expert_data_path, mesh)+'/save150'
			# params_path='{}/{}'.format(args.expert_data_path, mesh)

			# expert_policy = load_ddpg.load_policy(load_path, params_path)
			# env = gym.make("SawyerPushAndReachEnvEasy-v0",reward_type=args.reward_type)
			
			# print("Performing rollouts after training....")
			# roll, plot_data = rollout(env,
			# 	args.num_dagger_rollouts,
			# 	args.max_path_length,
			# 	policy,
			# 	expert_policy,
			# 	mesh = mesh, 
			# 	image_env=False)
			# env.close()
			# tf.get_variable_scope().reuse_variables()
			# data = append_paths(data, roll)
			
			pool = ThreadPool(processes=num_threads)
			async_results = [pool.apply_async(do_rollouts, (mesh, i, per_thread_rollouts, policy,)) for i in range(num_threads)]

			all_rollouts = []
			for i in range(num_threads):
				all_rollouts.append(async_results[i].get())

			for roll in all_rollouts:
				data = append_paths(data, roll)

			#for key in plotters.keys(): plotters[key].append(plot_data[key])


		# minro,maxro,meanro,meanfo= session.run([min_return_op,max_return_op,mean_return_op,mean_final_success_op],feed_dict=\
		# 	{min_return:np.min(plotters['min_return']),max_return:np.max(plotters['max_return']),mean_return:np.mean(plotters['mean_return']),\
		# 	mean_final_success:np.mean(plotters['mean_final_success'])})
		# set_writer.add_summary(minro,global_step=global_step)
		# set_writer.add_summary(maxro,global_step=global_step)
		# set_writer.add_summary(meanro,global_step=global_step)
		# set_writer.add_summary(meanfo,global_step=global_step)

		# for key in plotters.keys(): plotters[key].append(plot_data[key])

		if (i+1)%args.checkpoint_freq==0:
			savemodel(saver, session, checkpoint_dir_, i+1)

	plotting_data(plotters)
	session.__exit__()
	session.close()


def test(args):

	# Dictionary of values to plot
	plotters = {'min_return': [],
				'max_return': [],
				'mean_return': [],
				'mean_final_success': []}

	# Create environment
	env = gym.make("SawyerPushAndReachEnvEasy-v0",reward_type=args.reward_type)
	
	## Define policy network
	policy = XYZ_XYZ_Policy("dagger_xyz_xyz", env, hidden_sizes=[64, 64, 32])

	# Start session
	session = tfu.make_session(num_cpu=2)
	session.__enter__()

	session = tf.get_default_session()
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	session.run(init_op)

	ckpt = tf.train.get_checkpoint_state(args.checkpoint_path)
	print("checkpoint State ", ckpt)

	saver = tf.train.Saver()
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		print(("...found %s " % ckpt.model_checkpoint_path))
		print(ckpt_name)
		print(ckpt.model_checkpoint_path)
		saver.restore(session,ckpt.model_checkpoint_path)
	else:
		print("...ain't no full checkpoint here!")

	# Rollout policy
	for mesh in expert_list:
		print('testing {} '.format(mesh))
		change_env_to_use_correct_mesh(mesh)
		env = gym.make("SawyerPushAndReachEnvEasy-v0",reward_type=args.reward_type)
		
		_, stats = rollout(env,
				args.test_num_rollouts,
				args.max_path_length,
				policy,
				mesh = mesh, image_env=False, is_test=True)


		for key, value in enumerate(stats):
			print("{} : {}".format(value, stats[value]))

		for key in plotters.keys(): plotters[key].append(stats[key])

	plott = {'min_return': np.min(plotters['min_return']),
				'max_return': np.max(plotters['max_return']),
				'mean_return': np.mean(plotters['mean_return']),
				'mean_final_success': np.mean(plotters['mean_final_success'])}
	for key, value in enumerate(plott):
		print("{} : {}".format(value, plott[value]))

	session.close()

def plotting_data(plotters):

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
		plt.plot(range(len(plotters[key])), plotters[key])
		plt.title(key)
	plt.tight_layout()
	plt.savefig('metrics.png', dpi=300)
	plt.close()

def savemodel(saver, sess, checkpoint_dir, step):
	model_name = "baseline.model"
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	saver.save(sess,
			   os.path.join(checkpoint_dir, model_name),
			   global_step=step)
	print(("Saved a checkpoint: %s/%s-%d" % (checkpoint_dir, model_name, step)))

if __name__ == '__main__':
	args = parse_args()
	if args.test_policy:
		test(args)
	else:
		main(args)