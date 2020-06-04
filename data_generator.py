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
import gc
multiworld.register_all_envs()

IMSIZE = 64
# Reads from multiworld and generates datasets
class DataGenerator():

    def __init__(self, env="SawyerPushAndReachEnvEasy-v0",
                        reward_type='puck_success',
                        experts=["mug1"]):

        gym_env = gym.make(env, reward_type=reward_type)
        self.camera_space = {'dist_low': 0.7,'dist_high': 1.5,'angle_low': 0,'angle_high': 180,'elev_low': -180,'elev_high': -90}
        #self.camera_space = {'dist_low': 2.0,'dist_high': 2.0,'angle_low': 90,'angle_high': 90,'elev_low': -150,'elev_high': -150}
        self.num_cameras = hyp.S
        self.env = ImageEnv(
                wrapped_env=gym_env,
                imsize= IMSIZE, #64,
                normalize=True,
                camera_space=self.camera_space,
                init_camera=(lambda x: init_multiple_cameras(x, self.camera_space)),
                num_cameras=self.num_cameras,
                depth=True,
                cam_info=True,
                reward_type='wrapped_env',
                flatten=False
            )
        self.experts = experts

    def reinitialize_env(self, env="SawyerPushAndReachEnvEasy-v0", reward_type='puck_success'):
        gym_env = gym.make(env, reward_type=reward_type)
        self.env = ImageEnv(
                wrapped_env=gym_env,
                imsize= IMSIZE, #64,
                normalize=True,
                camera_space=self.camera_space,
                init_camera=(lambda x: init_multiple_cameras(x, self.camera_space)),
                num_cameras=self.num_cameras,
                depth=True,
                cam_info=True,
                reward_type='wrapped_env',
                flatten=False
            )

    def generate_data(self, num_rollouts, basedir, dataset_type="train"):
        identifier = time.strftime("%H%M%S")
        basic_info_file = "basic_info.pkl"
        tfrecord_file = "data"
        init_rollouts = num_rollouts
        for mesh in self.experts:
            num_rollouts = init_rollouts
            print("Generating data for expert {}".format(mesh))
            change_env_to_use_correct_mesh(mesh)
            
            # Needs to be done to update the mesh being used in the environment
            self.reinitialize_env()

            expert_load_path = args.expert_path + "/" + mesh + "/" + args.expert_ckpt
            expert_params_path = args.expert_path + "/" + mesh
            expert_policy = load_ddpg.load_policy(expert_load_path, expert_params_path)

            print("Num Rollouts ", args.num_rollouts)
            print("Max Path Length ", args.max_path_length)
            run = 1
            data = None
            while num_rollouts > 0:
                if num_rollouts > 100:
                    roll, _ = rollout(self.env,
                                100,
                                args.max_path_length,
                                expert_policy,
                                mesh=mesh, 
                                data_gen=True)
                    num_rollouts -= 100
                else:
                    roll, _ = rollout(self.env,
                                num_rollouts,
                                args.max_path_length,
                                expert_policy,
                                mesh=mesh, 
                                data_gen=True)

                    num_rollouts = 0

                if data is not None:
                    data = append_paths(data, roll)
                else:
                    data = roll
                    
                data_dict, basic_info = get_inputs(data, data['puck_zs'])
                record_files = convert_to_tfrecords(data, data_dict, tfrecord_filename="{}/{}/{}_{}_{}_{}".format(basedir, dataset_type, mesh, tfrecord_file, run, identifier), mesh=mesh)
                with open("{}/{}".format(basedir, dataset_type + "_records.txt"), 'a+') as f:
                    f.writelines(record_files)
                run += 1
                gc.collect()
            #import pdb; pdb.set_trace()

        if dataset_type == 'train':
            with open("{}/{}".format(basedir, basic_info_file), 'wb') as pkl:
                pickle.dump(basic_info, pkl)
            print("Dumped basic info into file!")

        print("Done!")
        #np.save('expert_data_{}.npy'.format(args.env), data)

def main(args):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.save_dir + "_" + timestamp)
    os.makedirs(args.save_dir + "_" + timestamp + "/train")
    os.makedirs(args.save_dir + "_" + timestamp + "/val")
    os.makedirs(args.save_dir + "_" + timestamp + "/test")

    experts = ['car2', 'eyeglass','headphones']
    #experts = ['knife2', 'mouse', 'mug1'] #['car2', 'eyeglass','headphones', 'knife2', 'mouse', 'mug1'] 
    #experts = ['book', 'car2', 'eyeglass', 'hamet', 'headphones', 'keyboard',  'knife2',  'mouse',  'mug1'] 
    generator = DataGenerator(experts=experts)
    generator.generate_data(num_rollouts=args.num_rollouts, basedir=args.save_dir + "_" + timestamp, dataset_type="train", )
    generator.generate_data(num_rollouts=args.num_rollouts*0.2, basedir=args.save_dir + "_" + timestamp, dataset_type="test")
    generator.generate_data(num_rollouts=args.num_rollouts*0.2, basedir=args.save_dir + "_" + timestamp, dataset_type="val")

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
    parser.add_argument('--save_dir', type=str, default='sawyer_3Dtensor_mujoco_{}'.format(IMSIZE))
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    main(args)