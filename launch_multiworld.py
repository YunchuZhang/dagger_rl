import gym
import multiworld
from multiworld.envs.mujoco.cameras import init_multiple_cameras
from multiworld.core.image_env import ImageEnv
import numpy as np 

from utils import change_env_to_use_correct_mesh, change_env_to_rescale_mesh

multiworld.register_all_envs()
# camera_space = {'dist_low': 0.75,'dist_high': 0.75,'angle_low': 180,'angle_high': 180,'elev_low': -90,'elev_high': -90}
# num_cameras = 4

mesh="keyboard"
change_env_to_use_correct_mesh(mesh)
change_env_to_rescale_mesh(mesh, scale="0.5 0.5 0.5")
env = gym.make("SawyerPushAnd__init__ReachEnvEasy-v0", reward_type='puck_success', z_rotation_angle=0)

# env = ImageEnv(
#                 wrapped_env=env,
#                 imsize=64,s
#                 normalize=True,
#                 camera_space=camera_space,
#                 init_camera=(lambda x: init_multiple_cameras(x, camera_space)),
#                 num_cameras=num_cameras,
#                 depth=True,
#                 cam_info=True,
#                 reward_type='wrapped_env',
#                 flatten=False
#             )
env.reset()
while True:
 	env.render()
 	env.step([])
env.close()