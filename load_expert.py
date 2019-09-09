import os
import json
import pickle
import gym
from softlearning.environments.utils import get_environment_from_params, get_environment
from softlearning.policies.utils import get_policy_from_variant
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import init_multiple_cameras

def get_environment_from_params_custom(environment_params):
    universe = environment_params['universe']
    task = environment_params['task']
    domain = environment_params['domain']
    # st()
    environment_kwargs_gym = environment_params.get('kwargs', {}).copy()
    if "map3D" in environment_kwargs_gym:
      environment_kwargs_gym.pop("map3D")
    if "observation_keys" in environment_kwargs_gym:
      environment_kwargs_gym.pop("observation_keys")
    env = gym.make(f"{domain}-{task}",**environment_kwargs_gym)

    camera_space={'dist_low': 0.7,'dist_high': 1.5,'angle_low': 0,'angle_high': 180,'elev_low': -180,'elev_high': -90}

    env_n = ImageEnv(
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

    environment_kwargs = environment_params.get('kwargs', {}).copy()
    environment_kwargs["env"] = env_n
    return get_environment(universe, domain, task, environment_kwargs)


def get_policy(checkpoint_path):
    checkpoint_path = checkpoint_path.rstrip('/')
    experiment_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(experiment_path, 'params.json')
    with open(variant_path, 'r') as f:
        variant = json.load(f)

    environment_params = (
        variant['environment_params']['evaluation']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])
    evaluation_environment = get_environment_from_params(environment_params)

    policy = (get_policy_from_variant(variant, evaluation_environment, Qs=[None]))
    training_environment = get_environment_from_params_custom(environment_params)

    return policy, training_environment