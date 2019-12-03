import argparse
import numpy as np
import os
import multiworld
import gym
import time
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import init_multiple_cameras
from utils import change_env_to_use_correct_mesh
from rollouts import rollout, append_paths

import load_ddpg
import pickle

from process_mujoco_inputs import get_inputs, convert_to_tfrecords
import hyperparams as hyp

multiworld.register_all_envs()

# Reads from multiworld and generates datasets
class DataGenerator():

    def __init__(self, env="SawyerPushAndReachEnvEasy-v0",
                        reward_type='puck_success',
                        experts=["mug1"]):

        gym_env = gym.make(env, reward_type=reward_type)
        self.camera_space = {'dist_low': 0.7,'dist_high': 1.5,'angle_low': 0,'angle_high': 180,'elev_low': -180,'elev_high': -90}

        num_cameras = hyp.S
        self.env = ImageEnv(
                wrapped_env=gym_env,
                imsize=64,
                normalize=True,
                camera_space=self.camera_space,
                init_camera=(lambda x: init_multiple_cameras(x, self.camera_space)),
                num_cameras=num_cameras,
                depth=True,
                cam_info=True,
                reward_type='wrapped_env',
                flatten=False
            )
        self.experts = experts

    def generate_data(self, num_rollouts, dataset_type="train"):
        data = None
        for mesh in self.experts:
            print("Generating data for expert {}".format(mesh))
            change_env_to_use_correct_mesh(mesh)
            expert_load_path = args.expert_path + "/" + mesh + "/" + args.expert_ckpt
            expert_params_path = args.expert_path + "/" + mesh
            expert_policy = load_ddpg.load_policy(expert_load_path, expert_params_path)

            print("Num Rollouts ", args.num_rollouts)
            print("Max Path Length ", args.max_path_length)
            roll, _ = rollout(self.env,
                        num_rollouts,
                        args.max_path_length,
                        expert_policy,
                        mesh=mesh, 
                        data_gen=True)
            if data is not None:
                data = append_paths(data, roll)
            else:
                data = roll

        data_dict, basic_info = get_inputs(data, data['puck_zs'])
        basic_info_file = "basic_info.pkl"
        tfrecord_file = "data"
        if dataset_type == 'train':
            with open("{}_{}/{}".format(args.save_dir, timestamp, basic_info_file), 'wb') as pkl:
                pickle.dump(basic_info, pkl)
            print("Dumped basic info into file!")
        record_files = convert_to_tfrecords(data, data_dict, tfrecord_filename="{}_{}/{}/{}".format(args.save_dir, timestamp, dataset_type, tfrecord_file))
        with open("{}_{}/{}".format(args.save_dir, timestamp, dataset_type + "_records.txt"), 'w') as f:
            f.writelines(record_files)
        print("Done!")
        #np.save('expert_data_{}.npy'.format(args.env), data)

def main(args):
    experts = ['car3','eyeglass'] #,'headphones','mouse','mug1']
    generator = DataGenerator(experts=experts)
    generator.generate_data(num_rollouts=args.num_rollouts, dataset_type="train")
    generator.generate_data(num_rollouts=args.num_rollouts, dataset_type="test")
    generator.generate_data(num_rollouts=args.num_rollouts, dataset_type="val")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        default='SawyerPushAndReachEnvEasy-v0',
                        help='Environment we are trying to run.')
    parser.add_argument('--max-path-length', '-l', type=int, default=50)
    parser.add_argument('--num-rollouts', '-n', type=int, default=1000)
    parser.add_argument('--expert-path', type=str, required=True)
    parser.add_argument('--expert-ckpt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='sawyer_3Dtensor_mujoco')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.save_dir + "_" + timestamp)
    os.makedirs(args.save_dir + "_" + timestamp + "/train")
    os.makedirs(args.save_dir + "_" + timestamp + "/val")
    os.makedirs(args.save_dir + "_" + timestamp + "/test")
    main(args)