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
from tensorflow import keras
from tensorflow.keras import layers

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

## Define environment
expert_list = ['car2']#, 'eyeglass','headphones']#, 'knife2', 'mouse', 'mug1']

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
						required=True,
						help='Path to some initial expert data collected.')
	parser.add_argument('--max-path-length', '-l', type=int, default=50)
	parser.add_argument('--num-rollouts', '-n', type=int, default=500)
	parser.add_argument('--num-dagger-rollouts', '-n-dgr', type=int, default=10)
	parser.add_argument('--test-num-rollouts', '-tn', type=int, default=20)
	parser.add_argument('--num-iterations', type=int, default=50)
	parser.add_argument('--mb_size', type=int, default=8)
	parser.add_argument('--checkpoint_freq', type=int, default=20)
	parser.add_argument('--reward_type', type=str, default='puck_success')
	parser.add_argument('--test-policy', type=bool, default=True)
	args = parser.parse_args()

	return args

def main(args):

	# Dictionary of values to plot
	plotters = {'min_return': [],
				'max_return': [],
				'mean_return': [],
				'mean_final_success': []}


	name = "baseline_3obj_imitation"
	log_dir_ = os.path.join("logs_sawyer_baseline_dagger", name)
	checkpoint_dir_ = os.path.join("ckpt_sawyer_baseline_imitation", name)
	set_writer = tf.summary.FileWriter(log_dir_ + '/train', None)

	## Define expert
	env = gym.make("SawyerPushAndReachEnvEasy-v0",reward_type=args.reward_type)

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
	
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(2)
  ])

  optimizer = tf.keras.optimizers.Adam(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

def test(args):

	## Define environment
	if args.mesh is not None: change_env_to_use_correct_mesh(args.mesh)

	# Dictionary of values to plot
	plotters = {'min_return': [],
				'max_return': [],
				'mean_return': [],
				'mean_final_success': []}

	# Create environment
	env = gym.make("SawyerPushAndReachEnvEasy-v0",reward_type=args.reward_type)
	
	## Define policy network
	policy = XYZ_XYZ_Policy("dagger_xyz_xyz", env)

	# Start session
	session = tfu.make_session(num_cpu=40)
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
	if args.test_policy:
		test(args)