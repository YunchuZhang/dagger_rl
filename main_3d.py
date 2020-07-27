import argparse
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import pickle
import numpy as np
import tqdm
import wandb
from glob import glob
from xml.etree import ElementTree as et

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import gym
gym.logger.set_level(40)
import load_ddpg
import policies.tf_utils as tfu
from gym.envs.robotics.image_env import ImageEnv
from gym.envs.robotics.camera import init_multiple_cameras
from policies.tensor_xyz_policy import Tensor_XYZ_Policy
from rollouts import rollout, append_paths
from utils import make_env, get_latest_checkpoint

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def get_ckpt_name(mesh):
	if mesh.find('_') == -1:
		name = 'v7_push1_'+ mesh[:5]
	else:
		name = 'v7_push1_'+ mesh[:5] + '_' +mesh[mesh.index('_')+1:]
	return name
def change_data(path,filenames,idx):
	storage = []	
	for state in idx[:args.mb_size]:
		with open(path + '/' + filenames[state], 'rb') as f:
			pkl = pickle.loads(f.read())
		storage.append(( pkl["observation"], pkl['desired_goal'],\
		pkl['achieved_goal'],pkl['image_observation'],pkl["depth_observation"],pkl['cam_info_observation'],\
		pkl['actions'],pkl['terminals'],pkl['obj_sizes'],pkl['puck_zs']))


	observation, observation_with_orientation, desired_goal, achieved_goal, image_observation,\
	depth_observation, cam_info_observation, actions, terminals, obj_sizes,puck_zs = \
	[], [], [], [], [] ,[] ,[] ,[] ,[] ,[] ,[]

	for i in range(args.mb_size): 
		a,c,d,e,f,g,h,a2,b2,c2 = storage[i]
		observation.append(np.array(a, copy=False))
		# observation_with_orientation.append(np.array(b, copy=False))
		desired_goal.append(np.array(c, copy=False))
		achieved_goal.append(np.array(d, copy=False))
		image_observation.append(np.array(e, copy=False))
		depth_observation.append(np.array(f, copy=False))
		cam_info_observation.append(np.array(g, copy=False))
		actions.append(np.array(h, copy=False))
		terminals.append(np.array(a2, copy=False))
		obj_sizes.append(np.array(b2, copy=False))
		puck_zs.append(np.array(c2, copy=False))
	total_data = {}
	total_data.update({'observation': np.array(observation)})
	# total_data.update({'observation_with_orientation': np.array(observation_with_orientation)})
	total_data.update({'desired_goal': np.array(desired_goal)})
	total_data.update({'achieved_goal': np.array(achieved_goal)})
	total_data.update({'image_observation': np.array(image_observation)})
	total_data.update({'depth_observation': np.array(depth_observation)})
	total_data.update({'cam_info_observation': np.array(cam_info_observation)})
	total_data.update({'actions': np.array(actions)})
	total_data.update({'terminals': np.array(terminals)})
	total_data.update({'obj_sizes': np.array(obj_sizes)})
	total_data.update({'puck_zs': np.array(puck_zs)})
		
	return total_data



def parse_args():
	parser = argparse.ArgumentParser()

	# environment
	parser.add_argument('--env',
			type=str,
			default='FetchPush-v1',
			help='Environment we are trying to run.')
	parser.add_argument('--goal_type',
			type=str,
			default='xyz',
			choices=['xyz', '3d'])
	parser.add_argument('--obs_type',
			type=str,
			default='xyz',
			choices=['xyz', '3d'])
	parser.add_argument('--base_xml_path',
			type=str,
			default='gym/envs/robotics/assets/fetch/push.xml',
			help='path to base xml of the environment relative to gym directory')
	parser.add_argument('--task_config_path',
			type=str,
			default='/projects/katefgroup/quantized_policies/quan_meshes_jy/push_v7_train_aug.yaml',
			help='path to task config relative to current directory')

	# policy
	parser.add_argument('--checkpoint_path',
			type=str,
			help='Path to the checkpoint.')
	parser.add_argument('--expert_data_path',
			default=None,
			type=str,
			help='Path to some initial expert data collected.')

	# training
	parser.add_argument('--max_path_length', '-l', type=int, default=50)
	parser.add_argument('--num_rollouts', '-n', type=int, default=10)
	parser.add_argument('--test_num_rollouts', '-tn', type=int, default=100)
	parser.add_argument('--num_iterations', type=int, default=50)
	parser.add_argument('--mb_size', type=int, default=8)
	parser.add_argument('--checkpoint_freq', type=int, default=5)
	parser.add_argument('--test_policy', action='store_true')
	parser.add_argument('--reward_type', type=str, default='sparse')
	parser.add_argument('--rollout_interval', type=int, default=4)

	# learning rate
	parser.add_argument('--learning_rate', type=float, default=1e-3)
	parser.add_argument('--decay_steps', type=float, default=20000)
	parser.add_argument('--decay_rate', type=float, default=0.75)

	# logging
	parser.add_argument('--prefix', type=str, default='default')
	parser.add_argument('--policy_name', type=str, default='dagger_tensor_xyz')
	parser.add_argument('--log_img_interval', type=int, default=5)
	parser.add_argument('--num_visualized_episodes', type=int, default=3,
						help='number of visualized episodes per mesh rollout')
	parser.add_argument('--wandb', action='store_true')

	args = parser.parse_args()

	return args


def main(args):
	# expert_list = sorted([x.split('/')[-2] for x in glob(os.path.join(args.expert_data_path, '*/'))])
	expert_list =  [
	 '2e9a0e216c08293d1395331ebe4786cd',
	# '1a1dcd236a1e6133860800e6696b8284_r135_x12962',
 '2e9a0e216c08293d1395331ebe4786cd_r090',
 # '2acbb7959e6388236d068062d5d5809b_r045',
 '1a48d03a977a6f0aeda0253452893d75_r135_x11444',
 # '1a97f3c83016abca21d0de04f408950f_x12094',
 '1a48d03a977a6f0aeda0253452893d75_r045_x10796',
 # '2b5a333c1a5aede3b5449ea1678de914_r135',
 '1ccd676efa14203e4b08e5e8ba35fb4_r045_x10255',
 # '7e984643df66189454e185afc91dc396_r090_x10023',
 '1a48d03a977a6f0aeda0253452893d75_r135',
 # '1b9c827d2109f4b2c98c13d6112727de_r090_x13450',
 '6dc3773e131f8001e76bc197b3a3ffc0_x07988',
 # '1a0bc9ab92c915167ae33d942430658c_r090_x08307',
 '4fdb0bd89c490108b8c8761d8f1966ba_x11504',
 # '1bc5d303ff4d6e7e1113901b72a68e7c_x13760',
 '1ccd676efa14203e4b08e5e8ba35fb4_r135_x07285',
 # '7a0a4269578ee741adba2821eac2f4fa_x13843',
 '1a1dcd236a1e6133860800e6696b8284_r090_x09325',
 # '1b64b36bf7ddae3d7ad11050da24bb12_x08388',
 '1a48d03a977a6f0aeda0253452893d75_r045_x07440',
 # '2d8bb293b226dcb678c34bb504731cb9_x12874',
 '2a46fe04fdd20bfdbda733a39f84326d_r090_x12369',
 # '63c10cfd6f0ce09a241d076ab53023c1_x11987',
 '1b9c827d2109f4b2c98c13d6112727de_r045_x11211',
 # '27f58201df188ce0c76e1e2d1feb4ae_x12597',
 '1c9568bc37d53a61c98c13d6112727de_r045_x10092',
 # '1e0ecacc02c5584d41cefd10ce5d6cc0_x08516',
 '2acbb7959e6388236d068062d5d5809b_r045_x11167',
 # '2b5a333c1a5aede3b5449ea1678de914_x09527',
 '381db800b87b5ff88616812464c86290_x13748',
 # '2b5a333c1a5aede3b5449ea1678de914',
 'a3c9dcaada9e04d09061da204e7c463c_x08954',
 # '1ccd676efa14203e4b08e5e8ba35fb4_x09824',
 '7e984643df66189454e185afc91dc396_r135_x12189',
 # '72d9a4452f6d58b5a0eb5a85db887292_r090_x07631',
 '1c38ca26c65826804c35e2851444dc2f_x10135',
 # '381db800b87b5ff88616812464c86290_x13829',
 '90198c0aaf0156ce764e2db342c0e628',
 # '2b5a333c1a5aede3b5449ea1678de914_x12509',
 '7e984643df66189454e185afc91dc396_x12384',
 # '3dc5a6d79ed591bda709dec9a148b2fe_r090',
 '1a1dcd236a1e6133860800e6696b8284_r090',
 # '4ad7d03773d3767c2bc52a80abcabb17_r045_x12330',
 '2b28e2a5080101d245af43a64155c221',
 # '1ccd676efa14203e4b08e5e8ba35fb4_r090_x08492',
 'a3c9dcaada9e04d09061da204e7c463c_x11139',
 # '4ad7d03773d3767c2bc52a80abcabb17_r045_x11039',
 '2e9a0e216c08293d1395331ebe4786cd_r045_x10666',
 # '1ccd676efa14203e4b08e5e8ba35fb4_x08769',
 # '2acbb7959e6388236d068062d5d5809b_r135_x13568',
 '1b74500dc5d2a6547c02d07bab7b395c_r135']
	data_path = '/home/yunchuz/fetchtemp/dagger_rl/data'
	# Dictionary of values to plot
	plotters = {'min_return': [],
				'max_return': [],
				'mean_return': [],
				'mean_final_success': []}

	log_dir_ = os.path.join(os.getcwd(), "logs/tensor_xyz/", args.prefix)
	checkpoint_dir_ = os.path.join(log_dir_, 'ckpt')
	set_writer = tf.summary.FileWriter(log_dir_ + '/train', None)

	## initialize wandb
	if args.wandb:
		wandb.init(name='dagger_rl.tensor_xyz.{}'.format(args.prefix),
				   config=args,
				   entity="katefgroup",
				   project="quantize",
				   tags=['dagger_rl', 'tensor_xyz'],
				   job_type='test' if args.test_policy else 'training',
				   sync_tensorboard=True)

	## Define expert
	env = make_env(args.env,
				   base_xml_path=args.base_xml_path,
				   obj_name=expert_list[0],
				   task_config_path=args.task_config_path,
				   reward_type=args.reward_type)
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
	policy = Tensor_XYZ_Policy(args.policy_name, env)

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

	lr = tf.train.exponential_decay(learning_rate=args.learning_rate,
									global_step=step,
									decay_steps=args.decay_steps,
									decay_rate=args.decay_rate,
									staircase=False)

	# Exclude ddpg network from gradient computation
	freeze_patterns = []
	freeze_patterns.append("ddpg")
	freeze_patterns.append("feat")

	loss = tf.reduce_mean(tf.squared_difference(policy.ac, act))
	train_vars = tf.contrib.framework.filter_variables(tf.trainable_variables(),
			exclude_patterns=freeze_patterns)
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
	init = True
	for mesh in expert_list:
		print('generating {} data'.format(mesh))

		# define expert
		name = get_ckpt_name(mesh)
		params_path = os.path.join(args.expert_data_path,name,'logs')
		load_path = get_latest_checkpoint(os.path.join(args.expert_data_path,name,'ckpt'))
		expert_policy = load_ddpg.load_policy(load_path, params_path)
		env = make_env(args.env,
					   base_xml_path=args.base_xml_path,
					   obj_name=mesh,
					   task_config_path=args.task_config_path,
					   reward_type=args.reward_type)
		camera_space={'dist_low': 1.,'dist_high': 1.6,'angle_low': 135,'angle_high': -135,'elev_low': -160,'elev_high': -90}
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

		# Collect initial data
		if init is True:
			data, _ = rollout(env,
					args.num_rollouts,
					args.max_path_length,
					expert_policy,
					mesh = mesh,
					is_init_data=True,
					base_xml_path = args.base_xml_path,
					task_config_path = args.task_config_path,
					)
			# np.save('expert_data_{}.npy'.format(args.env), data)
			onlyfiles = next(os.walk(data_path))[2]
			totalnum = len(onlyfiles)
			print('---------')
			print("before",totalnum)

			for counter in range(data['achieved_goal'].shape[0]):
				#save the dagger expert trajectories 
				expert_data = {'observation': np.array(data['observation'][counter]),
					# 'observation_with_orientation': np.array(data['observation_with_orientation'][counter]),
					'desired_goal': np.array(data['desired_goal'][counter]),
					'achieved_goal': np.array(data['achieved_goal'][counter]),
					'image_observation':np.array(data['image_observation'][counter]),
					'depth_observation':np.array(data['depth_observation'][counter]),
					'cam_info_observation':np.array(data['cam_info_observation'][counter]),
					'actions':np.array(data['actions'][counter]),
					'terminals':np.array(data['terminals'][counter]),
					'obj_sizes':np.array(data['obj_sizes'][counter]),
					'puck_zs':np.array(data['puck_zs'][counter]),
				}
				#print('saving'+'{:d}'.format(counter)+'.pkl')
				with open(os.path.join(data_path, 'state' + "{:d}".format(totalnum+counter) + '.pkl'), 'wb') as f:
					pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
			onlyfiles = next(os.walk(data_path))[2]
			totalnum = len(onlyfiles)
			print('---------')
			print("after",totalnum)
			init = False
		else:
			data, _ = rollout(env,
					args.num_rollouts,
					args.max_path_length,
					expert_policy,
					mesh = mesh,
					base_xml_path = args.base_xml_path,
					task_config_path = args.task_config_path,
					)
			# roll = append_paths(roll, data)
			print('---------')
			print("before",totalnum)
			for counter in range(data['achieved_goal'].shape[0]):
				#save the dagger expert trajectories 
				expert_data = {'observation': np.array(data['observation'][counter]),
					# 'observation_with_orientation': np.array(data['observation_with_orientation'][counter]),
					'desired_goal': np.array(data['desired_goal'][counter]),
					'achieved_goal': np.array(data['achieved_goal'][counter]),
					'image_observation':np.array(data['image_observation'][counter]),
					'depth_observation':np.array(data['depth_observation'][counter]),
					'cam_info_observation':np.array(data['cam_info_observation'][counter]),
					'actions':np.array(data['actions'][counter]),
					'terminals':np.array(data['terminals'][counter]),
					'obj_sizes':np.array(data['obj_sizes'][counter]),
					'puck_zs':np.array(data['puck_zs'][counter]),
				}
				#print('saving'+'{:d}'.format(counter)+'.pkl')
				with open(os.path.join(data_path, 'state' + "{:d}".format(totalnum+counter) + '.pkl'), 'wb') as f:
					pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
			onlyfiles = next(os.walk(data_path))[2]
			totalnum = len(onlyfiles)
			print('---------')
			print("after",totalnum)
		env.close()
		tf.get_variable_scope().reuse_variables()

	## Start training

	# Start for loop
	global_step = 0

	for i in tqdm.tqdm(range(args.num_iterations)):
		# Parse dataset for supervised learning
		filenames = os.listdir(data_path)
		num_samples = len(filenames)
		# print('num_samples',num_samples)
		idx = np.arange(num_samples)
		np.random.shuffle(idx)
		for j in range(num_samples // args.mb_size):
			np.random.shuffle(idx)
			data = change_data(data_path,filenames,idx)
			feed = policy.train_process_observation(data)
			act_train = data['actions']
			feed.update({act:act_train})

			loss, _ = session.run([loss_op,opt], feed_dict=feed)
			log_this = np.mod(global_step, 5000) == 0
			if log_this:
				results = session.run(policy.map3D.summary, feed)
				set_writer.add_summary(results, global_step)
			set_writer.add_summary(loss, global_step=global_step)
			global_step = global_step + 1
		# Generate some new dagger data after every few training iterations
		if i >= 5:
			if (i + 1) % args.rollout_interval == 0:
			# if i > 1:
				# Perform rollouts
				for mesh_ix, mesh in enumerate(expert_list):
					mesh_name = mesh if len(mesh) < 32 else 'obj{:2d}-{}'.format(mesh_ix, mesh[:4])
					print('Generating rollouts for mesh {}...'.format(mesh))

					# define expert
					name = get_ckpt_name(mesh)
					params_path = os.path.join(args.expert_data_path,name,'logs')
					load_path = get_latest_checkpoint(os.path.join(args.expert_data_path,name,'ckpt'))
					expert_policy = load_ddpg.load_policy(load_path, params_path)

					# init environment
					env = make_env(args.env,
								   base_xml_path=args.base_xml_path,
								   obj_name=mesh,
								   task_config_path=args.task_config_path,
								   reward_type=args.reward_type)
					camera_space={'dist_low': 1.,'dist_high': 1.6,'angle_low': 135,'angle_high': -135,'elev_low': -160,'elev_high': -90}

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

					should_render = (i // args.rollout_interval) % args.log_img_interval == 0

					data, plot_data = rollout(env,
							args.num_rollouts,
							args.max_path_length,
							policy,
							expert_policy,
							mesh = mesh,
							render=should_render,
							num_visualized_episodes=args.num_visualized_episodes,
							base_xml_path = args.base_xml_path,
							task_config_path = args.task_config_path,
						)

					env.close()

					# log scalars
					if args.wandb:
						wandb.log({'individual/mean_rew_{}'.format(mesh_name): plot_data['mean_return'],
							'individual/success_rate_{}'.format(mesh_name): plot_data['success_rate']})

					# log images if needed
					if should_render and args.wandb:
						vis_videos = np.array(plot_data['images']).transpose([0, 1, 4, 2, 3])
						wandb.log({"rollout_{}".format(mesh_name): wandb.Video(vis_videos, fps=5, format='mp4')})

					tf.get_variable_scope().reuse_variables()
					# data = append_paths(data, roll)
					print('---------')
					print("before",totalnum)
					for counter in range(data['achieved_goal'].shape[0]):
						#save the dagger expert trajectories 
						expert_data = {'observation': np.array(data['observation'][counter]),
							# 'observation_with_orientation': np.array(data['observation_with_orientation'][counter]),
							'desired_goal': np.array(data['desired_goal'][counter]),
							'achieved_goal': np.array(data['achieved_goal'][counter]),
							'image_observation':np.array(data['image_observation'][counter]),
							'depth_observation':np.array(data['depth_observation'][counter]),
							'cam_info_observation':np.array(data['cam_info_observation'][counter]),
							'actions':np.array(data['actions'][counter]),
							'terminals':np.array(data['terminals'][counter]),
							'obj_sizes':np.array(data['obj_sizes'][counter]),
							'puck_zs':np.array(data['puck_zs'][counter]),
						}
						#print('saving'+'{:d}'.format(counter)+'.pkl')
						with open(os.path.join(data_path, 'state' + "{:d}".format(totalnum+counter) + '.pkl'), 'wb') as f:
							pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
					onlyfiles = next(os.walk(data_path))[2]
					totalnum = len(onlyfiles)
					print('---------')
					print("after",totalnum)
					for key in plotters.keys(): plotters[key].append(plot_data[key])

				minro,maxro,meanro,meanfo= session.run([min_return_op,max_return_op,mean_return_op,mean_final_success_op],feed_dict=\
						{min_return:np.min(plotters['min_return']),max_return:np.max(plotters['max_return']),mean_return:np.mean(plotters['mean_return']),\
						mean_final_success:np.mean(plotters['mean_final_success'])})
				set_writer.add_summary(minro,global_step=global_step)
				set_writer.add_summary(maxro,global_step=global_step)
				set_writer.add_summary(meanro,global_step=global_step)
				set_writer.add_summary(meanfo,global_step=global_step)

		if (i+1)%args.checkpoint_freq==0:
			savemodel(saver, session, checkpoint_dir_, i+1)

	plotting_data(plotters)
	session.__exit__()
	session.close()
import yaml
from addict import Dict
def load_yaml(filename):
	with open(filename, 'r') as f:
		content = yaml.load(f, Loader=yaml.Loader)
	return content
def test(args):
	filename = "tasks/all.yaml"
	config = Dict(load_yaml(filename))
	expert_list = []
	for k in config['objs'].keys():
		expert_list.append(k)
	# expert_list = [
	# 'pen',
	# '583a67819f58209b2f3f67dd919744dd',
	# '381db800b87b5ff88616812464c86290',
	# 'a73d531b90965199e5f6d587dbc348b5',
	# '5ef0c4f8c0884a24762241154bf230ce',
	# '6e884701bfddd1f71e1138649f4c219'
	# ]
	# Dictionary of values to plot
	plotters = {'min_return': [],
				'max_return': [],
				'mean_return': [],
				'mean_final_success': []}

	# Create environment
	env = make_env(args.env,
				   base_xml_path=args.base_xml_path,
				   obj_name=expert_list[0],
				   task_config_path=args.task_config_path,
				   reward_type=args.reward_type)
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
	policy = Tensor_XYZ_Policy(args.policy_name, env)

	# Start session
	session = tfu.make_session(num_cpu=40)
	session.__enter__()

	policy.map3D.finalize_graph()

	ckpt = tf.train.get_checkpoint_state(args.checkpoint_path)
	saver = tf.train.Saver()
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		print(("...found %s " % ckpt.model_checkpoint_path))
		saver.restore(session, os.path.join(args.checkpoint_path, ckpt_name))
	else:
		print("...ain't no full checkpoint here!")
	# Rollout policy
	for mesh in expert_list:
		print('testing {} '.format(mesh))
		env = make_env(args.env,
					   base_xml_path=args.base_xml_path,
					   obj_name=mesh,
					   task_config_path=args.task_config_path,
					   reward_type=args.reward_type)
		camera_space={'dist_low': 1.,'dist_high': 1.6,'angle_low': 135,'angle_high': -135,'elev_low': -160,'elev_high': -90}
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
				mesh = mesh,
				num_visualized_episodes=args.num_visualized_episodes,
				base_xml_path = args.base_xml_path,
				task_config_path = args.task_config_path,
			)

		env.close()

		print('mesh: {} '.format(mesh))
		for key, value in enumerate(stats):
			if key == 'images':
				continue
			print("{} : {}".format(value, stats[value]))

		for key in plotters.keys(): plotters[key].append(stats[key])

	plott = {'min_return': np.min(plotters['min_return']),
			 'max_return': np.max(plotters['max_return']),
			 'mean_return': np.mean(plotters['mean_return']),
			 'mean_final_success': np.mean(plotters['mean_final_success'])}
	for key, value in enumerate(plott):
		print("{} : {}".format(value, plott[value]))
	# plotting_data(plott)
	session.close()


def plotting_data(plotters):
	# plot results
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
	model_name = "tensor_xyz.model"
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
