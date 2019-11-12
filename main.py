import argparse
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import pickle
import numpy as np
import tqdm
from xml.etree import ElementTree as et

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import multiworld
import gym
import load_ddpg
import tf_utils as tfu

from multiworld.envs.mujoco.cameras import init_multiple_cameras
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
	parser.add_argument('--max-path-length', '-l', type=int, default=50)
	parser.add_argument('--num-rollouts', '-n', type=int, default=10)
	parser.add_argument('--test-num-rollouts', '-tn', type=int, default=20)
	parser.add_argument('--num-iterations', type=int, default=50)
	parser.add_argument('--mb_size', type=int, default=8)
	parser.add_argument('--checkpoint_freq', type=int, default=5)

	args = parser.parse_args()

	return args

def main(args):

	## Define environment
	expert_list = ['mouse','headphones']
	if args.mesh is not None: change_env_to_use_correct_mesh(args.mesh)

	# Dictionary of values to plot
	plotters = {'min_return': [],
				'max_return': [],
				'mean_return': [],
				'mean_final_success': []}


	name = "baseline_2objs"
	log_dir_ = os.path.join("logs_mujoco_offline", name)
	checkpoint_dir_ = os.path.join("dagger_ckpts", name)
	set_writer = tf.summary.FileWriter(log_dir_ + '/train', None)

	## Define expert
	env = gym.make("SawyerPushAndReachEnvEasy-v0",reward_type='puck_success')

	## Define policy network
	policy = XYZ_XYZ_Policy("dagger_xyz_xyz", env)

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
									staircase=True)

	# Exclude map3D network from gradient computation
	freeze_patterns = []
	freeze_patterns.append("feat")

	loss = tf.reduce_mean(tf.squared_difference(policy.ac, act))
	train_vars = tf.contrib.framework.filter_variables( tf.trainable_variables(),
														exclude_patterns=freeze_patterns)
	# tf.get_variable_scope().reuse_variables()
	opt = tf.train.AdamOptimizer(learning_rate=lr, name="adam_optimizer").minimize(loss,
														var_list=train_vars,
														global_step=step)

	# Start session
	session = tfu.make_session(num_cpu=40)
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
	init = True
	for mesh in expert_list:
		print('generating {} data'.format(mesh))
		change_env_to_use_correct_mesh(mesh)
		## Define expert
		# /projects/katefgroup/sawyer_ddpg_weight
		load_path='/Users/apokle/Documents/goal_conditioned_policy/dagger_rl/ckpts/{}'.format(mesh)+'/save99'
		params_path='/Users/apokle/Documents/goal_conditioned_policy/dagger_rl/ckpts/{}'.format(mesh)

		expert_policy = load_ddpg.load_policy(load_path, params_path)
		env = gym.make("SawyerPushAndReachEnvEasy-v0", reward_type='puck_success')
		
		# Collect initial data
		if init is True:
			data, _ = rollout(env,
						args.num_rollouts,
						args.max_path_length,
						expert_policy,
						mesh = mesh, image_env=False)
			np.save('expert_data_{}.npy'.format(args.env), data)
			init = False
		else:
			roll, _ = rollout(env,
					args.num_rollouts,
					args.max_path_length,
					expert_policy,
					mesh = mesh, image_env=False)
			data = append_paths(data, roll)
		env.close()
		tf.get_variable_scope().reuse_variables()

	## Start training

	# Start for loop
	global_step = 0

	for i in tqdm.tqdm(range(args.num_iterations)):
		plotters = {'min_return': [],
				'max_return': [],
				'mean_return': [],
				'mean_final_success': []}
		# Parse dataset for supervised learning
		num_samples = data['achieved_goal'].shape[0]
		print('num_samples',num_samples)
		idx = np.arange(num_samples)
		np.random.shuffle(idx)
		for j in range(num_samples // args.mb_size):
			np.random.shuffle(idx)
			obs = policy.train_process_observation(data, idx[:args.mb_size])
			act_train = data['actions'][idx[:args.mb_size]]
			loss, _ = session.run([loss_op,opt], feed_dict={ob:obs, act:act_train})
			set_writer.add_summary(loss, global_step=global_step)
			global_step = global_step + 1

		# Perform rollouts
		for mesh in expert_list:
			print('generating {} dagger data'.format(mesh))
			change_env_to_use_correct_mesh(mesh)
			## Define expert
			load_path='/Users/apokle/Documents/goal_conditioned_policy/dagger_rl/ckpts/{}'.format(mesh)+'/save99'
			params_path='/Users/apokle/Documents/goal_conditioned_policy/dagger_rl/ckpts/{}'.format(mesh)

			# if mesh =='mouse':
			# 	load_path='/projects/katefgroup/apokle/ckpts/{}'.format(mesh)+'/save99'

			# expert_policy, env = load_expert.get_policy(checkpoint_path)

			expert_policy = load_ddpg.load_policy(load_path, params_path)
			env = gym.make("SawyerPushAndReachEnvEasy-v0",reward_type='puck_success')
			
			# Collect initial data
			roll, plot_data = rollout(env,
				args.num_rollouts,
				args.max_path_length,
				policy,
				expert_policy,
				mesh = mesh, 
				image_env=False)
			# import ipdb;ipdb.set_trace()
			env.close()
			tf.get_variable_scope().reuse_variables()
			data = append_paths(data, roll)

			for key in plotters.keys(): plotters[key].append(plot_data[key])


		minro,maxro,meanro,meanfo= session.run([min_return_op,max_return_op,mean_return_op,mean_final_success_op],feed_dict=\
			{min_return:np.min(plotters['min_return']),max_return:np.max(plotters['max_return']),mean_return:np.mean(plotters['mean_return']),\
			mean_final_success:np.mean(plotters['mean_final_success'])})
		set_writer.add_summary(minro,global_step=global_step)
		set_writer.add_summary(maxro,global_step=global_step)
		set_writer.add_summary(meanro,global_step=global_step)
		set_writer.add_summary(meanfo,global_step=global_step)

		# for key in plotters.keys(): plotters[key].append(plot_data[key])

		if (i+1)%args.checkpoint_freq==0:
			savemodel(saver, session, checkpoint_dir_, i+1)

	plotting_data(plotters)
	session.__exit__()
	session.close()


def test(args):

	## Define environment
	expert_list = ['knife2','car2','keyboard','mouse']
	if args.mesh is not None: change_env_to_use_correct_mesh(args.mesh)

	# Dictionary of values to plot
	plotters = {'min_return': [],
				'max_return': [],
				'mean_return': [],
				'mean_final_success': []}

	# Create environment
	env = gym.make("SawyerPushAndReachEnvEasy-v0",reward_type='puck_success')
	
	## Define policy network
	policy = XYZ_XYZ_Policy("dagger_xyz_xyz", env)

	# Start session
	session = tfu.make_session(num_cpu=40)
	session.__enter__()

	session = tf.get_default_session()
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	session.run(init_op)

	#session.run(tf.global_variables_initializer())

	checkpoint_path = "/Users/apokle/Documents/goal_conditioned_policy/dagger_rl/dagger_ckpts/baseline_2objs"

	# saver = tf.train.import_meta_graph(checkpoint_path+ "/minuet.model-0"+".meta")
	ckpt = tf.train.get_checkpoint_state(checkpoint_path)
	print("checkpoint State ", ckpt)
	saver = tf.train.Saver()
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		print(("...found %s " % ckpt.model_checkpoint_path))
		saver.restore(session, os.path.join(checkpoint_path, ckpt_name))
	else:
		print("...ain't no full checkpoint here!")

	# Rollout policy
	for mesh in expert_list:
		print('testing {} '.format(mesh))
		change_env_to_use_correct_mesh(mesh)
		env = gym.make("SawyerPushAndReachEnvEasy-v0",reward_type='puck_success')
		
		_, stats = rollout(env,
				args.test_num_rollouts,
				args.max_path_length,
				policy,
				mesh = mesh, image_env=False)


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
	main(args)