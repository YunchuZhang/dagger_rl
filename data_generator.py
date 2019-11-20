import argparse

import multiworld
import gym
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import init_multiple_cameras
from utils import change_env_to_use_correct_mesh
import load_ddpg


multiworld.register_all_envs()

# Reads from multiworld and generates datasets
class DataGenerator():

	def __init__(self, env="SawyerPushAndReachEnvEasy-v0",
						reward_type='puck_success',
						experts=["mug1"]):

		gym_env = gym.make(env, reward_type=reward_type)
		self.camera_space = {'dist_low': 0.7,'dist_high': 1.5,'angle_low': 0,'angle_high': 180,'elev_low': -180,'elev_high': -90}

		self.env = ImageEnv(
				wrapped_env=gym_env,
				imsize=64,
				normalize=True,
				camera_space=self.camera_space,
				init_camera=(lambda x: init_multiple_cameras(x, self.camera_space)),
				num_cameras=4,
				depth=True,
				cam_info=True,
				reward_type='wrapped_env',
				flatten=False
			)
		self.experts = experts

	def generate_data(self):
		for mesh in self.experts:
			print("Generating data for expert".format(mesh))
			change_env_to_use_correct_mesh(mesh)
			expert_load_path = args.expert_path + mesh + args.expert_ckpt
			expert_params_path = args.expert_path + mesh
			expert_policy = load_ddpg.load_policy(ckpt_path, params_path)

			data, _ = rollout(env,
						args.num_rollouts,
						args.max_path_length,
						expert_policy,
						mesh = mesh)

			import pdb; pdb.set_trace()
			np.save('expert_data_{}.npy'.format(args.env), data)

def main(args):
	experts = ['car2_new','car3','coffee_mug','eyeglass','headphones','keyboard','knife2_new','mouse','mug1']
	generator = DataGenerator(experts=experts)
	generator.generate_data()

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env',
						type=str,
						default='SawyerPushAndReachEnvEasy-v0',
						help='Environment we are trying to run.')
	parser.add_argument('--max-path-length', '-l', type=int, default=50)
	parser.add_argument('--num-rollouts', '-n', type=int, default=10^3)
	parser.add_argument('--expert-path', type=str, required=True)
	parser.add_argument('--expert-ckpt', type=str, required=True)
	args = parser.parse_args()	
	return args

if __name__=="__main__":
	args = parse_args()
	main(args)