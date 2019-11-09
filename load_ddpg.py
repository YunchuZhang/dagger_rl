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
	params = prepare_params(params)
	dims = config.configure_dims(params)
	policy = config.configure_ddpg(dims=dims, params=params, reuse = False,clip_return=clip_return)
	if load_path is not None:
		tf_util.load_variables(load_path)
		print("Successfully loaded a policy.")
	return policy

def change_env_to_use_correct_mesh(mesh):
	path_to_xml = os.path.join('/Users/zyc/Downloads/multiworld/multiworld/envs/assets/sawyer_xyz/sawyer_push_box.xml')
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
	physics_dict["keyboard"] =  ["3", "0.00016 0.00023 0.00008", "0.002 0.004 .0001" ]
	physics_dict["knife"] =  ["8", "0.00016 0.00023 0.00008", "0.0005 0.0004 .0001" ]
	physics_dict["pillow"] =  ["6", "0.00016 0.00023 0.00008", "0.5 0.4 .0001" ]
	physics_dict["bag22"] =  ["8", "0.00016 0.00023 0.00008", "0.01 0.01 .0001" ]
	physics_dict["knife2"] =  ["8", "0.00016 0.00023 0.00008", "0.0005 0.0004 .0001" ]

	#set parameters
	[x.attrib for x in root.iter('geom')][0]['mass'] = physics_dict[mesh][0]
	# [x.attrib for x in root.iter('inertial')][0]['diaginertia'] = physics_dict[mesh][1]
	[x.attrib for x in root.iter('geom')][0]['friction'] = physics_dict[mesh][2]

	tree.write(path_to_xml)

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
		# import ipdb;ipdb.set_trace()

		actions, _, _, _ = model.step(obs)

		obs, rew, done, _ = env.step(actions)
		# print(actions)
		episode_rew += rew[0] 
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