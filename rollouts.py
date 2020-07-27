import sys
import os
import cv2
import getpass
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from quantized_policies.pcp_utils.utils import config_from_yaml_file, get_gym_dir, get_root_dir
from quantized_policies.pcp_utils.mesh_object import MeshObject
from quantized_policies.pcp_utils.parse_task_files import generate_integrated_xml
from quantized_policies.pcp_utils.np_vis import compute_bonuding_box_from_obj_xml, get_bbox_attribs
EXPERT_KEYS = ['observation',
			   'desired_goal',
			   'achieved_goal',
			   ]

baseline_short_keys = ['observation',
					   'desired_goal',
					   'achieved_goal',
					   ]

# short_keys = ['observation',
# 			  'observation_with_orientation',
# 			  'observation_abs',
# 			  'desired_goal',
# 			  'desired_goal_abs',
# 			  'achieved_goal',
# 			  'achieved_goal_abs',
# 			  'image_observation',
# 			  'object_pose',
# 			  # 'object_pos',
# 			  # 'image_desired_goal',
# 			  # 'image_achieved_goal',
# 			  'depth_observation',
# 			  # 'depth_desired_goal',
# 			  'cam_info_observation',
# 			  # 'cam_info_goal',
# 			  ]
short_keys = ['observation', 
	'achieved_goal', 
	'desired_goal', 
	'gripper_to_rim', 
	'image_observation', 
	'depth_observation', 
	'cam_info_observation']
	# 'image_desired_goal', 
	# 'depth_desired_goal', 
	# 'cam_info_goal'
	# 'image_achieved_goal']




def evaluate_rollouts(paths):
	"""Compute evaluation metrics for the given rollouts."""
	total_returns = [path['rewards'].sum() for path in paths]
	episode_lengths = [len(p['rewards']) for p in paths]

	diagnostics = OrderedDict((
		('return-average', np.mean(total_returns)),
		('return-min', np.min(total_returns)),
		('return-max', np.max(total_returns)),
		('return-std', np.std(total_returns)),
		('episode-length-avg', np.mean(episode_lengths)),
		('episode-length-min', np.min(episode_lengths)),
		('episode-length-max', np.max(episode_lengths)),
		('episode-length-std', np.std(episode_lengths)),
		))

	return diagnostics

def convert_to_active_observation(x):
	flattened_observation = np.concatnate([
		x[key] for key in EXPERT_KEYS], axis=-1)
	return [flattened_observation[None]]

def return_stats(rewards, count_infos, goal_reach_percentage, images=None):
	return {
		'min_return': np.min(rewards),
		'max_return': np.max(rewards),
		'mean_return': np.mean(rewards),
		'mean_final_success': np.mean(count_infos),
		'success_rate': goal_reach_percentage,
		'images': images
	}

def rollout(env,
		num_rollouts,
		path_length,
		policy,
		expert_policy=None,
		mesh=None,
		image_env=True,
		is_test=False,
		is_init_data=False,
		scale=1.0,
		render=False,
		num_visualized_episodes=3,
		base_xml_path = None,
		task_config_path = None,
		):
	# configure observation keys
	env_keys = short_keys if image_env else baseline_short_keys

	if image_env:
		if not base_xml_path.startswith('/'):
			base_xml_path = os.path.join(get_gym_dir(), base_xml_path)

		if not task_config_path.startswith('/'):
			repo_dir = os.path.dirname(os.path.realpath(__file__))
			base_xml_path = os.path.join(repo_dir, base_xml_path)

		# generate modified xml
		objs_config = config_from_yaml_file(task_config_path)
		obj = MeshObject(objs_config['objs'][mesh], mesh)
		obj_xpos = env.env.sim.data.get_body_xpos('object0').copy()
		obj_xmat = env.env.sim.data.get_body_xmat('object0').copy()

		bbox_points = compute_bonuding_box_from_obj_xml(obj.obj_xml_file,obj_xpos,obj_xmat,obj.scale)
		bounds, center, extents = get_bbox_attribs(bbox_points)
		obj_size = np.repeat(np.max(extents), 3)
		puck_z = 0.01
	else:
		puck_z = 0.01

	if image_env:
		if mesh =='mug2' or mesh == 'mouse' or mesh == 'coffee_mug':
			obj_size = np.repeat(np.max(obj_size), 3)

	#import pdb; pdb.set_trace()
	# Check instance for DDPG
	# <baselines.her.ddpg.DDPG object at 0x7f70a8560e10>
	if str(policy).find('DDPG')!=-1:
		actor = policy.step
	else:
		actor = policy.act
		observation_converter = lambda x: x

	if expert_policy:
		assert str(expert_policy).find('DDPG')!=-1
		expert_actor = expert_policy.step

	paths = []
	rewards = []
	count_infos = []
	rollout_images = []
	img = 0
	count = 0
	img = env.render("rgb_array")
	img_resize = cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)
	img_gray = cv2.cvtColor(img_resize,cv2.COLOR_RGB2GRAY)
	res = np.zeros(img_gray.shape,dtype=np.float32)
	cv2.normalize(img_gray, res, 0, 1, cv2.NORM_MINMAX,dtype=cv2.CV_32F)
	img_f = res.reshape(64,64,1)
	pbar = tqdm(range(num_rollouts), desc='Rollout Mesh {}'.format(mesh))

	for rollout_ix in pbar:
		t = 0
		path = {key: [] for key in env_keys}
		path['img'] = []
		images = []
		infos = []
		observations = []
		actions = []
		terminals = []
		obj_sizes = []
		puck_zs = []
		observation = env.reset()
		# render initial image if needed
		should_render = render and rollout_ix < num_visualized_episodes
		if should_render:
			images.append(env.render(mode='rgb_array'))

		first_reward = True
		R = 0
		for t in range(path_length):
			if str(policy).find('DDPG')!=-1:
				action,_,_,_ = actor(observation)
			else:
				if image_env:
					action = actor(observation,obj_size,puck_z)
				else:
					action = actor(observation,img_f)

			if expert_policy:
				expert_action,_,_,_ = expert_actor(observation)
			else:
				expert_action = action

			observation, reward, terminal, info = env.step(action)
			if should_render:
				images.append(env.render(mode='rgb_array'))

			img = env.render("rgb_array")
			img_resize = cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)
			img_gray = cv2.cvtColor(img_resize,cv2.COLOR_RGB2GRAY)
			res = np.zeros(img_gray.shape,dtype=np.float32)
			cv2.normalize(img_gray, res, 0, 1, cv2.NORM_MINMAX,dtype=cv2.CV_32F)
			img_f = res.reshape(64,64,1)
			# scale observation
			# observation["observation"][5] *= scale

			for key in env_keys:
				path[key].append(observation[key])
			path['img'].append(img_f)
			actions.append(expert_action)
			terminals.append(terminal)

			if image_env:
				obj_sizes.append(obj_size)
			puck_zs.append(puck_z)

			infos.append(info)
			R += reward

			if reward == 0 and first_reward:
				count += 1
				first_reward = False

		# if R == path_length, the episode very likely failed
		# we can select to not append such paths
		# not performing this step for now 

		pbar.set_description("Rollout Mesh {} [rew={}, cnt={}/{}]".format(mesh, R, count, rollout_ix + 1))

		path = {key: np.stack(path[key], axis=0) for key in path.keys()}

		path['img'] = np.stack(path['img'], axis=0)
		path['actions'] = np.stack(actions, axis=0)
		path['terminals'] = np.stack(terminals, axis=0).reshape(-1, 1)
		if image_env:
			path['obj_sizes'] = np.stack(obj_sizes, axis=0)
		path['puck_zs'] = np.stack(puck_zs, axis=0).reshape(-1,1)
		# if isinstance(policy, GaussianPolicy) and len(path['terminals']) >= path_length:
		# 	continue
		# elif not isinstance(policy, GaussianPolicy) and len(path['terminals']) == 1:
		# 	continue
		rewards.append(R)
		count_infos.append(infos[-1]['is_success'])
		paths.append(path)
		if should_render:
			rollout_images.append(images)

	rollout_data = _clean_paths(paths)
	rollout_stats = return_stats(rewards, count_infos, count/num_rollouts,
			rollout_images if render else None)
	return rollout_data, rollout_stats


def _clean_paths(paths):
	"""Cleaning up paths to only contain relevant information like
	   observation, next_observation, action, reward, terminal.
	"""

	clean_paths = {key: np.concatenate([path[key] for path in paths]) for key in paths[0].keys()}

	return clean_paths

def append_paths(main_paths, paths):
	if main_paths is None or len(main_paths) == 0:
		return paths
	elif len(paths) == 0:
		return main_paths
	else:
		# append the rollouts obtained with already existing data
		paths = {key: np.vstack((main_paths[key], paths[key])) for key in main_paths.keys()}
		return paths
