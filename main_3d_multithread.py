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
import policies.tf_utils as tfu

from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import init_multiple_cameras
from policies.tensor_xyz_policy import Tensor_XYZ_Policy
from rollouts import rollout, append_paths
from softlearning.environments.gym.wrappers import NormalizeActionWrapper
from utils import change_env_to_use_correct_mesh

from multiprocessing.pool import ThreadPool, Pool

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

def do_rollouts(mesh, per_thread_rollouts=5):
	print('generating {} dagger data'.format(mesh))
	change_env_to_use_correct_mesh(mesh)
	## Define expert
	load_path='/projects/katefgroup/sawyer_ddpg_weight/{}'.format(mesh)+'/save150'
	params_path='/projects/katefgroup/sawyer_ddpg_weight/{}'.format(mesh)

	expert_policy = load_ddpg.load_policy(load_path, params_path)

	print("Initializing gym environment")
	env = gym.make("SawyerPushAndReachEnvEasy-v0",reward_type='puck_success')
	camera_space={'dist_low': 0.7,'dist_high': 1.5,'angle_low': 0,'angle_high': 180,'elev_low': -180,'elev_high': -90}
	print("Enclosing ImageEnv wrapper ")
	env = ImageEnv(
			wrapped_env=env,
			imsize=64,
			normalize=True,
			camera_space=camera_space,
			init_camera=(lambda x: init_multiple_cameras(x, camera_space)),
			num_cameras=4,#4 for training
			depth=True,
			cam_info=True,
			reward_type='wrapped_env',
			flatten=False
		)
	print("Starting rollouts")
	# Collect initial data
	roll, _ = rollout(env,
		per_thread_rollouts,
		args.max_path_length,
		policy,
		expert_policy,
		mesh = mesh)

	env.close()
	tf.get_variable_scope().reuse_variables()
	return roll

def print_squares(x):
	print("Square of {} : {}".format(x, x*x))

def main(args):

	## Define environment
	expert_list = ['car2', 'eyeglass','headphones']
	if args.mesh is not None: change_env_to_use_correct_mesh(args.mesh)

	# change_env_to_use_correct_mesh("mouse")
	# "/Users/zyc/Downloads/save200_mouse" 
	# load_path="/home/robertmu/checkpoint/ckpt_testone8/save200"

	# params_path="/home/robertmu/logs/back1"

	# Dictionary of values to plot
	plotters = {'min_return': [],
				'max_return': [],
				'mean_return': [],
				'mean_final_success': []}


	name = "correct_3objs_lr_0.001_decay_0.9_steps_2e4"
	log_dir_ = os.path.join("logs_sawyer_3dtensor_", name)
	checkpoint_dir_ = os.path.join("ckpts_3dtensor_", name)
	set_writer = tf.summary.FileWriter(log_dir_ + '/train', None)

	## Define expert
	env = gym.make("SawyerPushAndReachEnvEasy-v0",reward_type='puck_success')
	camera_space={'dist_low': 0.7,'dist_high': 1.5,'angle_low': 0,'angle_high': 180,'elev_low': -180,'elev_high': -90}

	env = ImageEnv(
			wrapped_env=env,
			imsize=64,
			normalize=True,
			camera_space=camera_space,
			init_camera=(lambda x: init_multiple_cameras(x, camera_space)),
			num_cameras=4,#4 for training
			depth=True,
			cam_info=True,
			reward_type='wrapped_env',
			flatten=False
		)

	# expert_policy, env = load_expert.get_policy(args.checkpoint_path)

	## Define policy network
	policy = Tensor_XYZ_Policy("dagger_tensor_xyz", env)

	act = tfu.get_placeholder(name="act",
							dtype=tf.float32,
							shape=[None, policy.act_dim])
	min_return = tfu.get_placeholder(dtype=tf.float32, shape=None, name="min_return")
	max_return = tfu.get_placeholder(dtype=tf.float32, shape=None, name="max_return")
	mean_return = tfu.get_placeholder(dtype=tf.float32, shape=None, name="mean_return")
	mean_final_success = tfu.get_placeholder(dtype=tf.float32, shape=None, name="mean_final_success")

	step = tf.Variable(0, trainable=False)

	lr = tf.train.exponential_decay(learning_rate = 0.001,
									global_step = step,
									decay_steps = 20000,
									decay_rate = 0.9,
									staircase=False)

	# Exclude ddpg network from gradient computation
	freeze_patterns = []
	freeze_patterns.append("ddpg")

	loss = tf.reduce_mean(tf.squared_difference(policy.ac, act))
	train_vars = tf.contrib.framework.filter_variables(tf.trainable_variables(), exclude_patterns=freeze_patterns)
	
	opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,
															var_list=train_vars,
															global_step=step)

	# Start session
	session = tfu.make_session(num_cpu=40)
	session.__enter__()

	policy.map3D.finalize_graph()
	# seperate with map_3d summary
	loss_op = tf.summary.scalar('loss', loss)

	with tf.variable_scope("policy_perf"):
		min_return_op = tf.summary.scalar('min_return', min_return)
		max_return_op = tf.summary.scalar('max_return', max_return)
		mean_return_op = tf.summary.scalar('mean_return', mean_return)
		mean_final_success_op = tf.summary.scalar('mean_final_success', mean_final_success)


	saver = tf.train.Saver()
	# Load expert policy
	data = None
	for mesh in expert_list:
		# print('generating {} data'.format(mesh))
		# change_env_to_use_correct_mesh(mesh)
		# ## Define expert
		# # /projects/katefgroup/sawyer_ddpg_weight
		# load_path='/projects/katefgroup/sawyer_ddpg_weight/{}'.format(mesh)+'/save150'
		# params_path='/projects/katefgroup/sawyer_ddpg_weight/{}'.format(mesh)

		# # if mesh =='mouse':
		# # 	load_path='/projects/katefgroup/apokle/ckpts/{}'.format(mesh)+'/save99'

		# # expert_policy, env = load_expert.get_policy(checkpoint_path)

		# expert_policy = load_ddpg.load_policy(load_path, params_path)
		# env = gym.make("SawyerPushAndReachEnvEasy-v0",reward_type='puck_success')
		# camera_space={'dist_low': 0.7,'dist_high': 1.5,'angle_low': 0,'angle_high': 180,'elev_low': -180,'elev_high': -90}

		# env =  ImageEnv(
		# 		wrapped_env=env,
		# 		imsize=64,
		# 		normalize=True,
		# 		camera_space=camera_space,
		# 		init_camera=(lambda x: init_multiple_cameras(x, camera_space)),
		# 		num_cameras=4,#4 for training
		# 		depth=True,
		# 		cam_info=True,
		# 		reward_type='wrapped_env',
		# 		flatten=False
		# 	)

		# # Collect initial data
		# roll, _ = rollout(env,
		# 		args.num_rollouts,
		# 		args.max_path_length,
		# 		expert_policy,
		# 		mesh = mesh)
		# data = append_paths(data, roll)
		# env.close()
		# tf.get_variable_scope().reuse_variables()


		num_threads = 5
		per_thread_rollouts = args.num_rollouts / num_threads

		pool = ThreadPool(processes=num_threads)
		async_results = [pool.apply_async(do_rollouts, (mesh, per_thread_rollouts,)) for i in range(num_threads)]

		print("Async Results : ")
		print(async_results)

		import pdb; pdb.set_trace()
		roll = [result.get() for result in async_results]

		data = append_paths(data, roll)
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
			feed = policy.train_process_observation(data, idx[:args.mb_size])
			act_train = data['actions'][idx[:args.mb_size]]
			feed.update({act:act_train})
			loss, _ = session.run([loss_op,opt], feed_dict=feed)
			log_this = np.mod(global_step, 500) == 0
			if log_this:
				results = session.run(policy.map3D.summary, feed)
				set_writer.add_summary(results, global_step)
			set_writer.add_summary(loss, global_step=global_step)
			global_step = global_step + 1

		if (i + 1) % 2 == 0:
			# Perform rollouts
			for mesh in expert_list:
				num_threads = 5
				per_thread_rollouts = args.num_rollouts / num_threads

				pool = ThreadPool(processes=num_threads)
				async_result = pool.apply_async(do_rollouts, (mesh, per_thread_rollouts,))

				roll = async_result.get()

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
	camera_space={'dist_low': 0.7,'dist_high': 1.5,'angle_low': 0,'angle_high': 180,'elev_low': -180,'elev_high': -90}

	env = ImageEnv(
			wrapped_env=env,
			imsize=64,
			normalize=True,
			camera_space=camera_space,
			init_camera=(lambda x: init_multiple_cameras(x, camera_space)),
			num_cameras=4,#4 for training
			depth=True,
			cam_info=True,
			reward_type='wrapped_env',
			flatten=False
		)

	## Define policy network
	policy = Tensor_XYZ_Policy("dagger_tensor_xyz", env)

	# Start session
	session = tfu.make_session(num_cpu=40)
	session.__enter__()

	policy.map3D.finalize_graph()

	# saver = tf.train.import_meta_graph(checkpoint_path+ "/minuet.model-0"+".meta")
	ckpt = tf.train.get_checkpoint_state(checkpoint_path)
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
		camera_space={'dist_low': 0.7,'dist_high': 1.5,'angle_low': 0,'angle_high': 180,'elev_low': -180,'elev_high': -90}

		env = ImageEnv(
				wrapped_env=env,
				imsize=64,
				normalize=True,
				camera_space=camera_space,
				init_camera=(lambda x: init_multiple_cameras(x, camera_space)),
				num_cameras=4,#4 for training
				depth=True,
				cam_info=True,
				reward_type='wrapped_env',
				flatten=False
			)	
		
		_, stats = rollout(env,
				args.test_num_rollouts,
				args.max_path_length,
				policy,
				mesh = mesh)


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
	model_name = "minuet.model"
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	saver.save(sess,
			   os.path.join(checkpoint_dir, model_name),
			   global_step=step)
	print(("Saved a checkpoint: %s/%s-%d" % (checkpoint_dir, model_name, step)))



if __name__ == '__main__':
	args = parse_args()
	main(args)
	# test(args)
	# _, env = load_expert.get_policy(args.checkpoint_path)
	# _,plot_data = test(env,args.num_rollouts,args.max_path_length)
	# plotting_data(plot_data)
