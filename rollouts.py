import numpy as np
import cv2
from softlearning.policies.gaussian_policy import GaussianPolicy
import sys
import getpass
# sys.path.append("/home/{}".format(getpass.getuser()))
# from discovery.backend.mujoco_online_inputs import get_inputs
import tensorflow as tf

EXPERT_KEYS = [	'observation',
				'desired_goal',
				'achieved_goal',
				]

baseline_short_keys = [	'observation',
				'desired_goal', 
				'achieved_goal']

short_keys = [	'observation',
				'observation_with_orientation',
				'observation_abs',
				'desired_goal',
				'desired_goal_abs',
				'achieved_goal',
				'achieved_goal_abs',
				'image_observation',
				'object_pose',
				# 'object_pos',
				# 'image_desired_goal', 
				# 'image_achieved_goal', 
				'depth_observation',
				# 'depth_desired_goal', 
				'cam_info_observation',
				]
				# 'cam_info_goal']



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
	flattened_observation = np.concatenate([
		x[key] for key in EXPERT_KEYS], axis=-1)
	return [flattened_observation[None]]

def return_stats(rewards, count_infos, goal_reach_percentage):
	return {'min_return': np.min(rewards),
			'max_return': np.max(rewards),
			'mean_return': np.mean(rewards),
			'mean_final_success': np.mean(count_infos),
			'success_rate': goal_reach_percentage}

def rollout(env,
			num_rollouts,
			path_length,
			policy,
			expert_policy=None,
			mesh = None, 
			image_env= True,
			is_test=False,
			is_init_data=False,
			scale=1.0):
	# env_keys = env.observation_space.spaces.keys()
	if image_env:
		env_keys = short_keys
	else:
		env_keys = baseline_short_keys

	if image_env:
		obj_size = env.sim.model.geom_size[env.sim.model.geom_name2id('puckbox')]
		obj_size = 2 * obj_size
		puck_z = env.init_puck_z + \
				env.sim.model.geom_pos[env.sim.model.geom_name2id('puckbox')][-1]
	else:
		puck_z = 0.01

	if image_env:
		if mesh =='mug2' or mesh == 'mouse' or mesh == 'coffee_mug':
			obj_size = np.repeat(np.max(obj_size), 3)

	#import pdb; pdb.set_trace()
	# Check instance for DDPG
	# <baselines.her.ddpg.DDPG object at 0x7f70a8560e10>
	if str(policy).find('DDPG')!=-1:
		# actions, _, _, _ = model.step(obs)
		# actor = policy.actions_np
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
	img = 0
	count = 0
	while len(paths) < (num_rollouts):
		print(len(paths))
		t = 0
		path = {key: [] for key in env_keys}
		images = []
		infos = []
		observations = []
		actions = []
		terminals = []
		obj_sizes = []
		puck_zs = []
		observation = env.reset()

		#print("Before rescaling obs ", observation["observation"])
		observation["observation"][5] *= scale
		#print("After rescaling obs ", observation["observation"])
		# cv2.imwrite('store/{}.png'.format(img),goal_img)
		# img = img + 1
		first_reward = True
		R = 0
		for t in range(path_length):
			# observation = observation_converter(observation)
			if str(policy).find('DDPG')!=-1:
				action,_,_,_ = actor(observation)
			else:
				if image_env:
					action = actor(observation,obj_size,puck_z)
				else:
					action = actor(observation)

			if expert_policy:
				# exp_observation = exp_observation_converter(observation)
				expert_action,_,_,_ = expert_actor(observation)
			else:
				expert_action = action

			observation, reward, terminal, info = env.step(action)
			
			#print("Before rescaling obs ", observation["observation"])
			observation["observation"][5] *= scale
			#print("After rescaling obs ", observation["observation"])

			# image = env.render(mode='rgb_array') #cv2 show image
			# cv2.imwrite('store/'+'{}_'.format(mesh)+'{}.png'.format(img),image)
			# img = img + 1

			for key in env_keys:
				path[key].append(observation[key])
			actions.append(expert_action)
			terminals.append(terminal)
			
			if image_env:
				obj_sizes.append(obj_size)
			puck_zs.append(puck_z)

			infos.append(info)
			R += reward

			if reward == 0 and first_reward:
				count += 1
				print("Episode Reward=", R, reward)
				first_reward = False

			# if terminal:
			# 	print('episode_rew={}'.format(R))

			# 	if isinstance(policy, GaussianPolicy):
			# 		policy.reset()
			# 	break
		#------------------------------------------------------------------------------
		# If R == path_length, the episode very likely failed; do not append such paths
		# Stopping this step for now --------------------------------------------------
		print("Reward ", R)
		#if is_test or (is_init_data and (R > -path_length)) or (len(actions) > 0):
		path = {key: np.stack(path[key], axis=0) for key in env_keys}
		path['actions'] = np.stack(actions, axis=0)
		path['terminals'] = np.stack(terminals, axis=0).reshape(-1,1)
		if image_env:
			path['obj_sizes'] = np.stack(obj_sizes, axis=0)
		path['puck_zs'] = np.stack(puck_zs, axis=0).reshape(-1,1)
		if isinstance(policy, GaussianPolicy) and len(path['terminals']) >= path_length:
			continue
		elif not isinstance(policy, GaussianPolicy) and len(path['terminals'])==1:
			continue
		rewards.append(R)
		count_infos.append(infos[-1]['is_success'])
		paths.append(path)

	# print('Minimum return: {}'.format(np.min(rewards)))
	# print('Maximum return: {}'.format(np.max(rewards)))
	# print('Mean return: {}'.format(np.mean(rewards)))
	# print('Mean final success: {}'.format(np.mean(count_infos)))
	print('Goal Reach Percentage: {}'.format(count/num_rollouts))
	return _clean_paths(paths), return_stats(rewards, count_infos, count/num_rollouts)

def _clean_paths(paths):
	"""Cleaning up paths to only contain relevant information like
	   observation, next_observation, action, reward, terminal.
	"""

	clean_paths = {key: np.concatenate([path[key] for path in paths]) for key in paths[0].keys()}

	return clean_paths

def append_paths(main_paths, paths):
	if main_paths is None or len(main_paths) == 0:
		return paths
	if len(paths) == 0:
		return main_paths
	""" Appending the rollouts obtained with already exisiting data."""
	paths = {key: np.vstack((main_paths[key], paths[key])) for key in main_paths.keys()}
	return paths
