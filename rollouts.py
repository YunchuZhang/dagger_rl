import numpy as np
from softlearning.policies.gaussian_policy import GaussianPolicy

EXPERT_KEYS = ['observation_with_orientation',
				'desired_goal',
				'achieved_goal',
				'state_observation',
				'state_desired_goal',
				'state_achieved_goal',
				'proprio_observation',
				'proprio_desired_goal',
				'proprio_achieved_goal']

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

def rollout(env,
			num_rollouts,
			path_length,
			policy,
			expert_policy=None):

	env_keys = env.observation_space.spaces.keys()
	# Check instance for softlearning
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
	while len(paths) < (num_rollouts):

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
			if isinstance(policy, GaussianPolicy):
				action = actor(observation)[0]
			else:
				action = actor(observation)
			if expert_policy:
				exp_observation = exp_observation_converter(observation)
				expert_action = expert_actor(exp_observation)[0]
			else:
				expert_action = action
			observation, reward, terminal, info = env.step(action)

			for key in env_keys:
				path[key].append(observation[key])
			actions.append(expert_action)
			terminals.append(terminal)
			infos.append(info)
			R += reward

			if terminal:
				
				if isinstance(policy, GaussianPolicy):
					policy.reset()
				break

		assert len(infos) == t + 1
		print("total_steps",t+1)


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
