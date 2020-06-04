import argparse

import tensorflow as tf 
import load_ddpg
import tqdm
from utils import change_env_to_use_correct_mesh
from rollouts import rollout, append_paths
##### Imports related to environment #############
import gym
import multiworld

multiworld.register_all_envs()

##### Imports for logging at wandb ##############
import wandb

meshes = ['car2', 'eyeglass','headphones', 'knife2', 'mouse', 'mug1']

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--expert_data_path',
						default=None,
						type=str,
						help='Path to some initial expert data collected.')
	parser.add_argument('--max-path-length', '-l', type=int, default=50)
	parser.add_argument('--num-rollouts', '-n', type=int, default=500)
	args = parser.parse_args()

	return args

def gather_rollout_stats(args):
	n_meshes = len(meshes)

	for src in range(n_meshes):
		src_mesh = meshes[src]
		# Load the policy corresponding to the source mesh
		load_path='{}/{}'.format(args.expert_data_path, src_mesh)+'/save150'
		params_path='{}/{}'.format(args.expert_data_path, src_mesh)

		expert_policy = load_ddpg.load_policy(load_path, params_path)

		for tgt in range(n_meshes):
			wandb.init(project="gather-rollout-stats")
			# Change the environment mesh to use target mesh
			tgt_mesh = meshes[tgt]
			change_env_to_use_correct_mesh(tgt_mesh)
			#initialize environments with this mesh
			env = gym.make('SawyerPushAndReachEnvEasy-v0')

			print("Starting rollouts for src {} tgt {} ".format(src_mesh, tgt_mesh))
			# perform rollouts to gather stats
			_, stats = rollout(env,
						args.num_rollouts,
						args.max_path_length,
						expert_policy,
						mesh = tgt_mesh, 
						image_env=False)

			wandb.log({'src':src_mesh, 'tgt':tgt_mesh, 'success_rate': stats['success_rate']})

def main(args):
	gather_rollout_stats(args)

if __name__=="__main__":
	args = parse_args()
	main(args)