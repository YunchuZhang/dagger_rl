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
import load_expert
import tf_utils as tfu

from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import init_multiple_cameras
from policies.tensor_xyz_policy import Tensor_XYZ_Policy
from rollouts import rollout, append_paths,test
from softlearning.environments.gym.wrappers import NormalizeActionWrapper

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

multiworld.register_all_envs()

def change_env_to_use_correct_mesh(mesh):
	path_to_xml = os.path.join('/home/robertmu/bc_robert/multiworld/multiworld/envs/assets/sawyer_xyz/sawyer_push_box.xml')
	tree = et.parse(path_to_xml)
	root = tree.getroot()
	[x.attrib for x in root.iter('geom')][0]['mesh']=mesh
	 #set the masses, inertia and friction in a plausible way

	physics_dict = {}
	physics_dict["printer"] =  ["6.0", ".00004 .00003 .00004", "1 1 .0001" ]
	physics_dict["mug1"] =  ["0.31", ".000000001 .0000000009 .0000000017", "0.008 0.008 .00001" ]
	physics_dict["mug2"] =  ["16.5", ".000001 .0000009 .0000017", "0.4 0.2 .00001" ]
	physics_dict["mug3"] =  ["0.33", ".000000001 .0000000009 .0000000017", "0.008 0.008 .00001" ]
	physics_dict["can1"] =  ["0.55", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
	physics_dict["car1"] =  ["0.2", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
	physics_dict["car2"] =  ["0.4", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
	physics_dict["car3"] =  ["5.5", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
	physics_dict["car4"] =  ["0.8", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
	physics_dict["car5"] =  ["2.0", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
	physics_dict["boat"] =  ["17.0", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
	physics_dict["bowl"] =  ["10", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
	physics_dict["bowl2"] =  ["1", ".00002 .00002 .00001", "0.2 0.2 .0001" ]
	physics_dict["bowl4"] =  ["0.7", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
	physics_dict["hat1"] =  ["0.2", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
	physics_dict["hat2"] =  ["0.4", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
	physics_dict["mouse"] = ["2.7", ".00027 .00025 .00016", "1.5 0.5 .000001"]
	physics_dict["book"] = ["10", ".00768 .01193 .00646", "3.5 2.5 .000001"]
	physics_dict["coffee_mug"] = ["21", ".0007 .0002 .0007", "0.35 0.25 .000001"]
	physics_dict["boat2"] =  ["6.0", ".00002 .00002 .00001", "0.2 0.2 .0001" ]
	physics_dict["headphones"] =  ["3", ".0012 .0039 .0029", "0.7 0.4 .0001" ]
	physics_dict["ball"] =  ["9", "0.000007 0.000007 0.000007", "0.0005 0.0004 .0001" ]
	physics_dict["eyeglass"] =  ["2.5", "0.00016 0.00023 0.00008", "0.0005 0.0004 .0001" ]
	physics_dict["plane"] =  ["5.5", "0.00016 0.00023 0.00008", "0.0005 0.0004 .0001" ]
	physics_dict["hamet"] =  ["12.5", "0.00016 0.00023 0.00008", "0.005 0.004 .001" ]
	physics_dict["clock"] =  ["3.5", "0.00016 0.00023 0.00008", "0.00005 0.00004 .00001" ]
	physics_dict["skate"] =  ["12", "0.00016 0.00023 0.00008", "0.6 0.4 .0001" ]
	physics_dict["bag1"] =  ["3", "0.00016 0.00023 0.00008", "0.005 0.004 .0001" ]
	physics_dict["bag2"] =  ["8", "0.00016 0.00023 0.00008", "0.01 0.01 .0001" ]
	physics_dict["keyboard"] =  ["5", "0.00016 0.00023 0.00008", "0.002 0.004 .0001" ]
	physics_dict["knife"] =  ["8", "0.00016 0.00023 0.00008", "0.0005 0.0004 .0001" ]
	physics_dict["pillow"] =  ["6", "0.00016 0.00023 0.00008", "0.5 0.4 .0001" ]

	#set parameters
	[x.attrib for x in root.iter('geom')][0]['mass'] = physics_dict[mesh][0]
	# [x.attrib for x in root.iter('inertial')][0]['diaginertia'] = physics_dict[mesh][1]
	[x.attrib for x in root.iter('geom')][0]['friction'] = physics_dict[mesh][2]

	tree.write(path_to_xml)

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
	parser.add_argument('--max-path-length', '-l', type=int, default=40)
	parser.add_argument('--num-rollouts', '-n', type=int, default=10)
	parser.add_argument('--test-num-rollouts', '-tn', type=int, default=20)
	parser.add_argument('--num-iterations', type=int, default=50)
	parser.add_argument('--mb_size', type=int, default=8)
	parser.add_argument('--checkpoint_freq', type=int, default=5)

	args = parser.parse_args()

	return args

def main(args):

	## Define environment
	expert_list = ['mug1','mouse','mug2','headphones','eyeglass','coffee_mug','car3','book','hamet','plane','pillow']
	if args.mesh is not None: change_env_to_use_correct_mesh(args.mesh)

	# env = gym.make(args.env)
	# env = NormalizeActionWrapper(env)
	# env = ImageEnv(env,
	# 		 imsize=64,
	# 		 normalize=True,
	# 		 init_camera=init_multiple_cameras,
	# 		 num_cameras=10,
	# 		 num_views=4,
	# 		 depth=True,
	# 		 cam_angles=True,
	# 		 reward_type="wrapped_env",
	# 		 flatten=False)

	# Dictionary of values to plot
	plotters = {'min_return': [],
				'max_return': [],
				'mean_return': [],
				'mean_final_success': []}


	name = "test9obj"
	log_dir_ = os.path.join("logs_mujoco_offline", name)
	checkpoint_dir_ = os.path.join("checkpoints", name)
	set_writer = tf.summary.FileWriter(log_dir_ + '/train', None)

	## Define expert
	expert_policy, env = load_expert.get_policy(args.checkpoint_path)

	## Define policy network
	policy = Tensor_XYZ_Policy("dagger_tensor_xyz", env)

	## Define DAGGER loss
	# goal_obs = tfu.get_placeholder(name="goal_obs",
	# 						dtype=tf.float32,
	# 						shape=[None, policy.state_obs_dim + policy.state_desired_dim])
	# crop = tfu.get_placeholder(name="crop",
	# 						dtype=tf.float32,
	# 						shape=[None, 16, 16, 8, 32])
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
	opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,
															var_list=train_vars,
															global_step=step)

	# Start session
	session = tfu.make_session(num_cpu=40)
	session.__enter__()

	# Load map3D network
	freeze_list = tf.contrib.framework.filter_variables(
		tf.trainable_variables(),
		include_patterns=freeze_patterns)

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
	init = True
	for mesh in expert_list:
		print('generating {} data'.format(mesh))
		change_env_to_use_correct_mesh(mesh)
		## Define expert
		checkpoint_path = '/projects/katefgroup/yunchu/{}'.format(mesh)+'48/checkpoint_1350/'
		if mesh =='mug2':
			checkpoint_path = '/projects/katefgroup/yunchu/{}'.format(mesh)+'48/checkpoint_1450/'
		expert_policy, env = load_expert.get_policy(checkpoint_path)
		# Load expert policy
		pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
		with open(pickle_path, 'rb') as f:
			picklable = pickle.load(f)

		expert_policy.set_weights(picklable['policy_weights'])
		with expert_policy.set_deterministic(True):
		
			# Collect initial data
			if init is True:
				data, _ = rollout(env,
							args.num_rollouts,
							args.max_path_length,
							expert_policy,
							mesh = mesh)
				np.save('expert_data_{}.npy'.format(args.env), data)
				init = False
			else:
				roll, _ = rollout(env,
						args.num_rollouts,
						args.max_path_length,
						expert_policy,
						mesh = mesh)
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
		num_samples = data['state_observation'].shape[0]
		print('num_samples',num_samples)
		idx = np.arange(num_samples)
		np.random.shuffle(idx)
		for j in range(num_samples // args.mb_size):
			np.random.shuffle(idx)
			feed = policy.train_process_observation(data, idx[:args.mb_size] ,env)
			act_train = data['actions'][idx[:args.mb_size]]
			feed.update({act:act_train})
			loss, _ = session.run([loss_op,opt], feed_dict=feed)
			log_this = np.mod(global_step, 500) == 0
			if log_this:
				results = session.run(policy.map3D.summary, feed)
				set_writer.add_summary(results, global_step)
			set_writer.add_summary(loss, global_step=global_step)
			global_step = global_step + 1

		# Perform rollouts
		for mesh in expert_list:
			print('generating {} dagger data'.format(mesh))
			change_env_to_use_correct_mesh(mesh)
			## Define expert
			checkpoint_path = '/projects/katefgroup/yunchu/{}'.format(mesh)+'48/checkpoint_1350/'
			if mesh =='mug2':
				checkpoint_path = '/projects/katefgroup/yunchu/{}'.format(mesh)+'48/checkpoint_1450/'
			expert_policy, env = load_expert.get_policy(checkpoint_path)
			# Load expert policy
			pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
			with open(pickle_path, 'rb') as f:
				picklable = pickle.load(f)

			expert_policy.set_weights(picklable['policy_weights'])

			with expert_policy.set_deterministic(True):
			# Collect initial data
				roll, plot_data = rollout(env,
					args.num_rollouts,
					args.max_path_length,
					policy,
					expert_policy,
					mesh = mesh)
				# import ipdb;ipdb.set_trace()
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
	expert_list = ['mug1','mouse','mug2','headphones','ball','book','eyeglass']
	if args.mesh is not None: change_env_to_use_correct_mesh(args.mesh)

	# Dictionary of values to plot
	plotters = {'min_return': [],
				'max_return': [],
				'mean_return': [],
				'mean_final_success': []}

	# Create environment
	_, env = load_expert.get_policy(args.checkpoint_path)

	## Define policy network
	policy = Tensor_XYZ_Policy("dagger_tensor_xyz", env)

	# Start session
	session = tfu.make_session(num_cpu=40)
	session.__enter__()

	policy.map3D.finalize_graph()
	checkpoint_path = "/home/robertmu/DAGGER_discovery/checkpoints/test7obj"
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
		checkpoint_path = '/projects/katefgroup/yunchu/{}'.format(mesh)+'48/checkpoint_1400/'
		_, env = load_expert.get_policy(checkpoint_path)	
		
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
