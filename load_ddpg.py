import json
import sys, os
import multiworld
import gym
multiworld.register_all_envs()
import numpy as np
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import init_multiple_cameras
import baselines.her.experiment.config as config
from baselines.common import tf_util
import tensorflow as tf
from xml.etree import ElementTree as et
from utils import change_env_to_use_correct_mesh
CACHED_ENVS = {}

def cached_make_env(make_env):
	if make_env not in CACHED_ENVS:
		env = make_env()
		CACHED_ENVS[make_env] = env
	return CACHED_ENVS[make_env]

def prepare_params(kwargs):
	# DDPG params
	ddpg_params = dict()
	env_name = kwargs['env_name']

	def make_env(subrank=None):
		env = gym.make(env_name)
		max_episode_steps = env._max_episode_steps
		env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
		return env

	kwargs['make_env'] = make_env
	tmp_env = cached_make_env(kwargs['make_env'])
	assert hasattr(tmp_env, '_max_episode_steps')

	kwargs['T'] = tmp_env._max_episode_steps

	kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
	kwargs['gamma'] = 1. - 1. / kwargs['T']
	if 'lr' in kwargs:
		kwargs['pi_lr'] = kwargs['lr']
		kwargs['Q_lr'] = kwargs['lr']
		del kwargs['lr']
	for name in ['buffer_size', 'hidden', 'layers',
				 'network_class',
				 'polyak',
				 'batch_size', 'Q_lr', 'pi_lr',
				 'norm_eps', 'norm_clip', 'max_u',
				 'action_l2', 'clip_obs', 'scope', 'relative_goals']:
		ddpg_params[name] = kwargs[name]
		kwargs['_' + name] = kwargs[name]
		del kwargs[name]
	kwargs['ddpg_params'] = ddpg_params

	return kwargs


def load_policy(load_path, params_path):
	with open(params_path + '/params.json') as f:
		params = json.load(f)
	clip_return=True

	with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
		params = prepare_params(params)
		dims = config.configure_dims(params)
		policy = config.configure_ddpg(dims=dims, params=params, reuse = False,clip_return=clip_return)
		if load_path is not None:
			tf_util.load_variables(load_path)
			print("Successfully loaded a policy.")
	return policy

def main():
	change_env_to_use_correct_mesh("hamet")
	# "/Users/zyc/Downloads/save200_mouse" 
	load_path="/Users/zyc/Downloads/save200"
	# "/Users/zyc/Downloads/save200_mouse" 
	
	# "/Users/zyc/Downloads"
	params_path="/Users/zyc/Downloads"
	# mesh = 'mouse'
	# load_path='/projects/katefgroup/apokle/ckpts/{}'.format(mesh)+'/save99'
	# params_path='/projects/katefgroup/apokle/ckpts/{}'.format(mesh)
	# "/Users/zyc/Downloads"
	

	model = load_policy(load_path, params_path)

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

	obs = env.reset()


	episode_rew = 0
	i = 0
	while True:

		actions, _, _, _ = model.step(obs)

		obs, rew, done, _ = env.step(actions)
		# print(actions)
		episode_rew += rew[-1] 
		env.render('wrapped')
		if done:
			print('episode_rew={}'.format(episode_rew))
			episode_rew = 0
			obs = env.reset()
		i+=1

	env.close()

	return model

if __name__ == '__main__':
	# while True:
	main()
		# tf.get_variable_scope().reuse_variables()