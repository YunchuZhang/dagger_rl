import numpy as np
import cv2
from softlearning.policies.gaussian_policy import GaussianPolicy
import sys
import getpass
# sys.path.append("/home/{}".format(getpass.getuser()))
# from discovery.backend.mujoco_online_inputs import get_inputs
import tensorflow as tf
import tf_utils as tfu
EXPERT_KEYS = ['observation_with_orientation',
				'desired_goal',
				'achieved_goal',
				'state_observation',
				'state_desired_goal',
				'state_achieved_goal',
				'proprio_observation',
				'proprio_desired_goal',
				'proprio_achieved_goal']
short_keys = ['observation_with_orientation', 
				'desired_goal', 
				'achieved_goal', 
				'state_observation', 
				'state_desired_goal', 
				'state_achieved_goal', 
				'proprio_observation', 
				'proprio_desired_goal', 
				'proprio_achieved_goal', 
				'image_observation', 
				# 'image_desired_goal', 
				# 'image_achieved_goal', 
				'depth_observation', 
				# 'depth_desired_goal', 
				'cam_info_observation']
				# 'cam_info_goal']
def evaluate_rollouts(paths):
	"""Compute evaluation metrics for the given rollouts."""

	import ipdb; ipdb.set_trace()
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

def return_stats(rewards, count_infos):
	return {'min_return': np.min(rewards),
			'max_return': np.max(rewards),
			'mean_return': np.mean(rewards),
			'mean_final_success': np.mean(count_infos)}



def test(env,num_rollouts,path_length):

	tf.reset_default_graph()
	session1 = tfu.make_session(num_cpu=40)
	session1.__enter__()

	session1.run(tf.global_variables_initializer())


	checkpoint_path = "/home/robertmu/DAGGER_discovery/checkpoints/dagger_tensor_xyz02"
	saver = tf.train.import_meta_graph(checkpoint_path+ "/minuet.model-0"+".meta")
	#print("i am reloading", tf.train.latest_checkpoint(checkpoint_path))
	saver.restore(session1,tf.train.latest_checkpoint(checkpoint_path))

	env_keys = env.observation_space.spaces.keys()
	observation_converter = lambda x: x

	paths = []
	rewards = []
	count_infos = []
	while len(paths) < (num_rollouts):
		import ipdb;ipdb.set_trace()

		t = 0
		path = {key: [] for key in env_keys}
		images = []
		infos = []
		observations = []
		actions = []
		terminals = []
		observation = env.reset()
		R = 0
		for t in range(path_length):
			observation = observation_converter(observation)
			ob = observation
			ob = {key: np.repeat(np.expand_dims(ob[key], axis=0),8, axis = 0) for key in ob.keys()}
			puck_z = env._env.env.init_puck_z + \
				env._env.env.sim.model.geom_pos[env._env.env.sim.model.geom_name2id('puckbox')][-1]
			batch_dict = get_inputs(ob,puck_z)
			goal_obs_train = np.hstack([ob['state_desired_goal'],ob['state_observation']])

			feed = {}

			feed.update({policy.rgb_camXs: batch_dict['rgb_camXs']})
			feed.update({policy.xyz_camXs: batch_dict['xyz_camXs']})
			feed.update({policy.pix_T_cams: batch_dict['pix_T_cams']})
			feed.update({policy.origin_T_camRs: batch_dict['origin_T_camRs']})
			feed.update({policy.origin_T_camXs: batch_dict['origin_T_camXs']})
			feed.update({policy.puck_xyz_camRs: batch_dict['puck_xyz_camRs']})
			feed.update({goal_obs: goal_obs_train})

			action = session.run([policy.ac], feed_dict=feed)

			observation, reward, terminal, info = env.step(action)

			for key in env_keys:
				path[key].append(observation[key])
			actions.append(action)
			terminals.append(terminal)

			infos.append(info)
			R += reward

			if terminal:
				break

		assert len(infos) == t + 1

		path = {key: np.stack(path[key], axis=0) for key in env_keys}
		path['actions'] = np.stack(actions, axis=0)
		path['terminals'] = np.stack(terminals, axis=0)
		if isinstance(policy, GaussianPolicy) and len(path['terminals']) >= path_length:
			continue
		elif not isinstance(policy, GaussianPolicy) and len(path['terminals'])==1:
			continue
		rewards.append(R)
		count_infos.append(infos[-1]['puck_success'])
		paths.append(path)

	return _clean_paths(paths), return_stats(rewards, count_infos)

def rollout(env,
			num_rollouts,
			path_length,
			policy,
			expert_policy=None,
			mesh = None):
	# env_keys = env.observation_space.spaces.keys()
	env_keys = short_keys
	obj_size = env._env.env.sim.model.geom_size[env._env.env.sim.model.geom_name2id('puckbox')]
	obj_size = 2 * obj_size
	puck_z = env._env.env.init_puck_z + \
			env._env.env.sim.model.geom_pos[env._env.env.sim.model.geom_name2id('puckbox')][-1]

	if mesh =='mug2' or mesh == 'mouse' or mesh == 'coffee_mug':
		obj_size = np.repeat(np.max(obj_size),3)

	# Check instance for softlearning
	# import ipdb;ipdb.set_trace()
	if isinstance(policy, GaussianPolicy):
		actor = policy.actions_np
		observation_converter = lambda x: convert_to_active_observation(x)
	else:
		actor = policy.act
		observation_converter = lambda x: x

	if expert_policy:
		assert isinstance(expert_policy, GaussianPolicy)
		expert_actor = expert_policy.actions_np
		exp_observation_converter = lambda x: convert_to_active_observation(x)

	paths = []
	rewards = []
	count_infos = []
	img = 0
	while len(paths) < (num_rollouts):

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
		# cv2.imwrite('store/{}.png'.format(img),goal_img)
		# img = img + 1
		R = 0
		for t in range(path_length):
			observation = observation_converter(observation)
			if isinstance(policy, GaussianPolicy):
				action = actor(observation)[0]
			else:
				action = actor(observation,obj_size,puck_z)
			if expert_policy:
				exp_observation = exp_observation_converter(observation)
				expert_action = expert_actor(exp_observation)[0]
			else:
				expert_action = action
			# print('action',action)
			observation, reward, terminal, info = env.step(action)
			# image = env.render(mode='rgb_array') #cv2 show image
			# cv2.imwrite('store/'+'{}_'.format(mesh)+'{}.png'.format(img),image)
			# img = img + 1

			for key in env_keys:
				path[key].append(observation[key])
			actions.append(expert_action)
			terminals.append(terminal)
			
			obj_sizes.append(obj_size)
			puck_zs.append(puck_z)

			infos.append(info)
			R += reward

			if terminal:

				if isinstance(policy, GaussianPolicy):
					policy.reset()
				break

		assert len(infos) == t + 1
		print('total_steps',t+1)


		path = {key: np.stack(path[key], axis=0) for key in env_keys}
		path['actions'] = np.stack(actions, axis=0)
		path['terminals'] = np.stack(terminals, axis=0)
		path['obj_sizes'] = np.stack(obj_sizes, axis=0)
		path['puck_zs'] = np.stack(puck_zs, axis=0).reshape(-1,1)
		if isinstance(policy, GaussianPolicy) and len(path['terminals']) >= path_length:
			continue
		elif not isinstance(policy, GaussianPolicy) and len(path['terminals'])==1:
			continue
		rewards.append(R)
		count_infos.append(infos[-1]['puck_success'])
		paths.append(path)

	# print('Minimum return: {}'.format(np.min(rewards)))
	# print('Maximum return: {}'.format(np.max(rewards)))
	# print('Mean return: {}'.format(np.mean(rewards)))
	# print('Mean final success: {}'.format(np.mean(count_infos)))

	return _clean_paths(paths), return_stats(rewards, count_infos)

def _clean_paths(paths):
	"""Cleaning up paths to only contain relevant information like
	   observation, next_observation, action, reward, terminal.
	"""

	clean_paths = {key: np.concatenate([path[key] for path in paths]) for key in paths[0].keys()}

	return clean_paths

def append_paths(main_paths, paths):
	""" Appending the rollouts obtained with already exisiting data."""
	paths = {key: np.vstack((main_paths[key], paths[key])) for key in main_paths.keys()}
	return paths
