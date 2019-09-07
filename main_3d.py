import argparse
import os

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
from rollouts import rollout, append_paths
from softlearning.environments.gym.wrappers import NormalizeActionWrapper

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

multiworld.register_all_envs()

def change_env_to_use_correct_mesh(mesh):
    path_to_xml = os.path.join('../multiworld/multiworld/envs/assets/sawyer_xyz/sawyer_push_box.xml')
    tree = et.parse(path_to_xml)
    root = tree.getroot()
    [x.attrib for x in root.iter('geom')][0]['mesh']=mesh
     #set the masses, inertia and friction in a plausible way

    physics_dict = {}
    physics_dict["printer"] =  ["6.0", ".00004 .00003 .00004", "1 1 .0001" ]
    physics_dict["mug1"] =  ["0.31", ".000000001 .0000000009 .0000000017", "0.008 0.008 .00001" ]
    physics_dict["mug2"] =  ["0.27", ".000000001 .0000000009 .0000000017", "0.008 0.008 .00001" ]
    physics_dict["mug3"] =  ["0.33", ".000000001 .0000000009 .0000000017", "0.008 0.008 .00001" ]
    physics_dict["can1"] =  ["0.55", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
    physics_dict["car1"] =  ["0.2", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
    physics_dict["car2"] =  ["0.4", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
    physics_dict["car3"] =  ["0.5", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
    physics_dict["car4"] =  ["0.8", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
    physics_dict["car5"] =  ["2.0", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
    physics_dict["boat"] =  ["7.0", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
    physics_dict["bowl1"] =  ["0.1", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
    physics_dict["bowl2"] =  ["0.3", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
    physics_dict["bowl4"] =  ["0.7", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
    physics_dict["hat1"] =  ["0.2", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
    physics_dict["hat2"] =  ["0.4", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]

    #set parameters
    [x.attrib for x in root.iter('inertial')][0]['mass'] = physics_dict[mesh][0]
    [x.attrib for x in root.iter('inertial')][0]['diaginertia'] = physics_dict[mesh][1]
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
	parser.add_argument('--max-path-length', '-l', type=int, default=100)
	parser.add_argument('--num-rollouts', '-n', type=int, default=10)
	parser.add_argument('--num-iterations', type=int, default=50)
	parser.add_argument('--mb_size', type=int, default=64)

	args = parser.parse_args()

	return args

def main(args):

	## Define environment
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


	name = "dagger_tensor_xyz"
	log_dir_ = os.path.join("logs_mujoco_offline", name)
	set_writer = tf.summary.FileWriter(log_dir_ + '/train', None)

	## Define expert
	expert_policy, env = load_expert.get_policy(args.checkpoint_path)

	## Define policy network
	policy = Tensor_XYZ_Policy("dagger_tensor_xyz", env)

	## Define DAGGER loss
	goal_obs = tfu.get_placeholder(name="goal_obs",
							dtype=tf.float32,
							shape=[None, policy.state_obs_dim + policy.state_desired_dim])
	crop = tfu.get_placeholder(name="crop",
							dtype=tf.float32,
							shape=[None, 16, 16, 8, 32])
	act = tfu.get_placeholder(name="act",
							dtype=tf.float32,
							shape=[None, policy.act_dim])
	loss = tf.reduce_mean(tf.squared_difference(policy.ac, act))
	opt = tf.train.AdamOptimizer().minimize(loss)

	

	# Start session
	session = tfu.make_session(num_cpu=40)
	session.__enter__()
	# session.run(tf.global_variables_initializer())

	# Load map3D network
	freeze_patterns = []
	freeze_patterns.append("feat")

	freeze_list = tf.contrib.framework.filter_variables(
		tf.trainable_variables(),
		include_patterns=freeze_patterns)


	policy.map3D.finalize_graph()
	# seperate with map_3d summary
	loss_op = tf.summary.scalar('loss', loss)

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
		np.save('expert_data_{}.npy'.format(args.env), data)
	else:
		data = np.load(args.expert_data_path, allow_pickle=True).item()
		roll, _ = rollout(env,
				args.num_rollouts,
				args.max_path_length,
				expert_policy)

	## Start training

	# Start for loop
	global_step = 0

	for i in tqdm.tqdm(range(args.num_iterations)):
		# print('\nIteration {} :'.format(i+1))
		# Parse dataset for supervised learning
		num_samples = data['state_observation'].shape[0]
		idx = np.arange(num_samples)
		np.random.shuffle(idx)
		for j in range(num_samples // args.mb_size):
			global_step = global_step + 1
			np.random.shuffle(idx)
			feat_train, goal_obs_train = policy.train_process_observation(data, idx[:args.mb_size])
			act_train = data['actions'][idx[:args.mb_size]]
			loss,_ = session.run([loss_op,opt], feed_dict={crop:feat_train, goal_obs:goal_obs_train, act:act_train})
			set_writer.add_summary(loss, global_step=global_step)
		
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
