import argparse
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import pickle
import numpy as np
import tqdm
import wandb
from glob import glob
from xml.etree import ElementTree as et

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
# import multiworld
import gym
gym.logger.set_level(40)
import load_ddpg
import policies.tf_utils as tfu

from policies.xyz_xyz_policy import XYZ_XYZ_Policy
from rollouts import rollout, append_paths
# from softlearning.environments.gym.wrappers import NormalizeActionWrapper
from utils import make_env, get_latest_checkpoint

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# multiworld.register_all_envs()


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--env',
            type=str,
            default='FetchPickAndPlace-v1',
            help='Environment we are trying to run.')
    parser.add_argument('--goal_type',
            type=str,
            default='xyz',
            choices=['xyz', '3d'])
    parser.add_argument('--obs_type',
            type=str,
            default='xyz',
            choices=['xyz', '3d'])
    parser.add_argument('--base_xml_path',
            type=str,
            default='gym/envs/robotics/assets/fetch/pick_and_place_kp30000.xml',
            help='path to base xml of the environment relative to gym directory')
    parser.add_argument('--task_config_path',
            type=str,
            default='tasks/all.yaml',
            help='path to task config relative to current directory')

    # policy
    parser.add_argument('--checkpoint_path',
            type=str,
            help='Path to the checkpoint.')
    parser.add_argument('--expert_data_path',
            default=None,
            type=str,
            help='Path to some initial expert data collected.')

    # training
    parser.add_argument('--max_path_length', '-l', type=int, default=50)
    parser.add_argument('--num_rollouts', '-n', type=int, default=500)
    parser.add_argument('--num_dagger_rollouts', '-n-dgr', type=int, default=50)
    parser.add_argument('--test_num_rollouts', '-tn', type=int, default=100)
    parser.add_argument('--num_iterations', type=int, default=50)
    parser.add_argument('--mb_size', type=int, default=16)
    parser.add_argument('--checkpoint_freq', type=int, default=5)
    parser.add_argument('--test_policy', action='store_true')
    parser.add_argument('--reward_type', type=str, default='sparse')
    parser.add_argument('--rollout_interval', type=int, default=2)

    # learning rate
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--decay_steps', type=float, default=20000)
    parser.add_argument('--decay_rate', type=float, default=0.9)

    # logging
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--policy_name', type=str, default='dagger_xyz_xyz')
    parser.add_argument('--log_img_interval', type=int, default=5)
    parser.add_argument('--num_visualized_episodes', type=int, default=3,
                        help='number of visualized episodes per mesh rollout')
    parser.add_argument('--wandb', action='store_true')

    args = parser.parse_args()

    return args
import yaml
from addict import Dict
def load_yaml(filename):
    with open(filename, 'r') as f:
        content = yaml.load(f, Loader=yaml.Loader)
    return content

def main(args):
    # expert_list = sorted([x.split('/')[-2] for x in glob(os.path.join(args.expert_data_path, '*/'))])
    filename = "tasks/push_small.yaml"
    config = Dict(load_yaml(filename))
    expert_list = []
    for k in config['objs'].keys():
        expert_list.append(k)
    # Dictionary of values to plot
    plotters = {'min_return': [],
                'max_return': [],
                'mean_return': [],
                'mean_final_success': []}

    log_dir_ = os.path.join(os.getcwd(), "logs/xyz_xyz/", args.prefix)
    checkpoint_dir_ = os.path.join(log_dir_, 'ckpt')
    set_writer = tf.summary.FileWriter(log_dir_ + '/train', None)

    ## initialize wandb
    if args.wandb:
        wandb.init(name='dagger_rl.xyz_xyz.{}'.format(args.prefix),
                   config=args,
                   entity="katefgroup",
                   project="quantize",
                   tags=['dagger_rl', 'xyz_xyz'],
                   job_type='test' if args.test_policy else 'training',
                   sync_tensorboard=True)

    ## Define expert
    env = make_env(args.env,
                   base_xml_path=args.base_xml_path,
                   obj_name=expert_list[0],
                   task_config_path=args.task_config_path,
                   reward_type=args.reward_type)

    ## Define policy network
    policy = XYZ_XYZ_Policy(args.policy_name, env, hidden_sizes=[64, 64, 32])
    ## Define DAGGER loss
    ob = tfu.get_placeholder(name="ob",
            dtype=tf.float32,
            shape=[None, policy.obs_dim])

    ## Define DAGGER loss
    act = tfu.get_placeholder(name="act",
            dtype=tf.float32,
            shape=[None, policy.act_dim])

    min_return = tfu.get_placeholder(dtype=tf.float32, shape=None, name="min_return")
    max_return = tfu.get_placeholder(dtype=tf.float32, shape=None, name="max_return")
    mean_return = tfu.get_placeholder(dtype=tf.float32, shape=None, name="mean_return")
    mean_final_success = tfu.get_placeholder(dtype=tf.float32, shape=None, name="mean_final_success")

    step = tf.Variable(0, trainable=False)

    lr = tf.train.exponential_decay(learning_rate=args.learning_rate,
                                    global_step=step,
                                    decay_steps=args.decay_steps,
                                    decay_rate=args.decay_rate,
                                    staircase=False)

    # Exclude ddpg network from gradient computation
    freeze_patterns = []
    freeze_patterns.append("ddpg")

    loss = tf.reduce_mean(tf.squared_difference(policy.ac, act))
    train_vars = tf.contrib.framework.filter_variables(tf.trainable_variables(),
            exclude_patterns=freeze_patterns)
    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,
            var_list=train_vars,
            global_step=step)

    # Start session
    session = tfu.make_session(num_cpu=2)
    session.__enter__()

    session = tf.get_default_session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    session.run(init_op)

    loss_op = tf.summary.scalar('loss', loss)

    with tf.variable_scope("policy_perf"):
        min_return_op = tf.summary.scalar('min_return', min_return)
        max_return_op = tf.summary.scalar('max_return', max_return)
        mean_return_op = tf.summary.scalar('mean_return', mean_return)
        mean_final_success_op = tf.summary.scalar('mean_final_success', mean_final_success)

    saver = tf.train.Saver()

    # Load expert policy
    init = True
    for mesh in expert_list:
        print('generating {} data'.format(mesh))

        # define expert
        params_path = os.path.join(args.expert_data_path,mesh,'logs')
        load_path = get_latest_checkpoint(os.path.join(args.expert_data_path,mesh,'ckpt'))

        expert_policy = load_ddpg.load_policy(load_path, params_path, obs_arg="bbox")
        env = make_env(args.env,
                       base_xml_path=args.base_xml_path,
                       obj_name=mesh,
                       task_config_path=args.task_config_path,
                       reward_type=args.reward_type)

        # Collect initial data
        if init is True:
            data, _ = rollout(env,
                    args.num_rollouts,
                    args.max_path_length,
                    expert_policy,
                    mesh = mesh,
                    image_env=False)
            np.save('expert_data_{}.npy'.format(args.env), data)
            init = False
        else:
            roll, _ = rollout(env,
                    args.num_rollouts,
                    args.max_path_length,
                    expert_policy,
                    mesh = mesh,
                    image_env=False)
            data = append_paths(data, roll)
        env.close()
        tf.get_variable_scope().reuse_variables()

    ## Start training

    # Start for loop
    global_step = 0

    for i in tqdm.tqdm(range(args.num_iterations)):
        # Parse dataset for supervised learning
        num_samples = data['achieved_goal'].shape[0]
        print('num_samples',num_samples)
        idx = np.arange(num_samples)
        np.random.shuffle(idx)
        for j in range(num_samples // args.mb_size):
            np.random.shuffle(idx)
            feed = policy.train_process_observation(data, idx[:args.mb_size])
            act_train = data['actions'][idx[:args.mb_size]]
            feed.update({act:act_train})
            loss, _ = session.run([loss_op,opt], feed_dict=feed)
            set_writer.add_summary(loss, global_step=global_step)
            global_step = global_step + 1

        # Generate some new dagger data after every few training iterations
        if (i + 1) % args.rollout_interval == 0:
            # Perform rollouts
            for mesh_ix, mesh in enumerate(expert_list):
                mesh_name = mesh if len(mesh) < 32 else 'obj{:2d}-{}'.format(mesh_ix, mesh[:4])
                print('Generating rollouts for mesh {}...'.format(mesh))

                # define expert
                # params_path = os.path.join(args.expert_data_path[:-6],'logs', mesh)
                # load_path = get_latest_checkpoint(os.path.join(args.expert_data_path, mesh))
                params_path = os.path.join(args.expert_data_path,mesh,'logs')
                load_path = get_latest_checkpoint(os.path.join(args.expert_data_path,mesh,'ckpt'))
                expert_policy = load_ddpg.load_policy(load_path, params_path)

                # init environment
                env = make_env(args.env,
                               base_xml_path=args.base_xml_path,
                               obj_name=mesh,
                               task_config_path=args.task_config_path,
                               reward_type=args.reward_type)

                should_render = (i // args.rollout_interval) % args.log_img_interval == 0
                roll, plot_data = rollout(env,
                        args.num_dagger_rollouts,
                        args.max_path_length,
                        policy,
                        expert_policy,
                        mesh = mesh,
                        image_env=False,
                        render=should_render,
                        num_visualized_episodes=args.num_visualized_episodes)

                env.close()

                # log scalars
                if args.wandb:
                    wandb.log({'individual/mean_rew_{}'.format(mesh_name): plot_data['mean_return'],
                        'individual/success_rate_{}'.format(mesh_name): plot_data['success_rate']})

                # log images if needed
                if should_render and args.wandb:
                    vis_videos = np.array(plot_data['images']).transpose([0, 1, 4, 2, 3])
                    wandb.log({"rollout_{}".format(mesh_name): wandb.Video(vis_videos, fps=5, format='mp4')})

                tf.get_variable_scope().reuse_variables()
                data = append_paths(data, roll)

                for key in plotters.keys(): plotters[key].append(plot_data[key])

            minro,maxro,meanro,meanfo= session.run([min_return_op,max_return_op,mean_return_op,mean_final_success_op],feed_dict=\
                    {min_return:np.min(plotters['min_return']),max_return:np.max(plotters['max_return']),mean_return:np.mean(plotters['mean_return']),\
                    mean_final_success:np.mean(plotters['mean_final_success'])})
            set_writer.add_summary(minro,global_step=global_step)
            set_writer.add_summary(maxro,global_step=global_step)
            set_writer.add_summary(meanro,global_step=global_step)
            set_writer.add_summary(meanfo,global_step=global_step)

        if (i+1)%args.checkpoint_freq==0:
            savemodel(saver, session, checkpoint_dir_, i+1)

    plotting_data(plotters)
    session.__exit__()
    session.close()


def test(args):
    # expert_list = [x.split('/')[-1] for x in glob(os.path.join(args.expert_data_path, '*/'))]
        ## initialize wandb
    if args.wandb:
        wandb.init(name='dagger_rl.xyz_xyz.{}'.format(args.prefix),
                   config=args,
                   entity="katefgroup",
                   project="quantize",
                   tags=['dagger_rl', 'xyz_xyz'],
                   job_type='test' if args.test_policy else 'training',
                   sync_tensorboard=True)
    filename = "tasks/all.yaml"
    config = Dict(load_yaml(filename))
    expert_list = []
    for k in config['objs'].keys():
        expert_list.append(k)
    # Dictionary of values to plot
    plotters = {'min_return': [],
                'max_return': [],
                'mean_return': [],
                'mean_final_success': []}

    # Create environment
    env = make_env(args.env,
                   base_xml_path=args.base_xml_path,
                   obj_name=expert_list[0],
                   task_config_path=args.task_config_path,
                   reward_type=args.reward_type)

    ## Define policy network
    policy = XYZ_XYZ_Policy(args.policy_name, env, hidden_sizes=[64, 64, 32])

    # Start session
    session = tfu.make_session(num_cpu=2)
    session.__enter__()

    session = tf.get_default_session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    session.run(init_op)

    ckpt = tf.train.get_checkpoint_state(args.checkpoint_path)
    saver = tf.train.Saver()
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(("...found %s " % ckpt.model_checkpoint_path))
        saver.restore(session, os.path.join(args.checkpoint_path, ckpt_name))
    else:
        print("...ain't no full checkpoint here!")

    # Rollout policy
    for mesh in expert_list:
        print('testing {} '.format(mesh))
        env = make_env(args.env,
                       base_xml_path=args.base_xml_path,
                       obj_name=mesh,
                       task_config_path=args.task_config_path,
                       reward_type=args.reward_type)
        should_render = True
        _, stats = rollout(env,
                args.test_num_rollouts,
                args.max_path_length,
                policy,
                mesh = mesh, image_env=False,render=should_render,
                num_visualized_episodes=args.num_visualized_episodes,is_test=True)
        env.close()

        # log scalars
        if args.wandb:
            wandb.log({'individual/success_rate_{}'.format(mesh): stats['success_rate']})

        # log images if needed
        if should_render and args.wandb:
            vis_videos = np.array(stats['images']).transpose([0, 1, 4, 2, 3])
            wandb.log({"rollout_{}".format(mesh): wandb.Video(vis_videos, fps=5, format='mp4')})

        for key, value in enumerate(stats):
            if value == 'images': continue
            print("{} : {}".format(value, stats[value]))

        for key in plotters.keys(): plotters[key].append(stats[key])

    plott = {'min_return': np.min(plotters['min_return']),
             'max_return': np.max(plotters['max_return']),
             'mean_return': np.mean(plotters['mean_return']),
             'mean_final_success': np.mean(plotters['mean_final_success'])}
    for key, value in enumerate(plott):
        print("{} : {}".format(value, plott[value]))

    session.close()


def plotting_data(plotters):
    # plot results
    color_list = ["#363737"]
    plt.figure(figsize=(4,4))
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 0.5
    plt.rcParams["font.sans-serif"] = "Helvetica"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["ytick.labelsize"] = "medium"
    plt.rcParams["xtick.labelsize"] = "medium"
    plt.rcParams["font.size"] = 8.3
    for i, key in enumerate(plotters.keys()):
        ax = plt.subplot(2,2,i+1)
        plt.plot(range(len(plotters[key])), plotters[key])
        plt.title(key)
    plt.tight_layout()
    plt.savefig('metrics.png', dpi=300)
    plt.close()


def savemodel(saver, sess, checkpoint_dir, step):
    model_name = "xyz_xyz.model"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver.save(sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)
    print(("Saved a checkpoint: %s/%s-%d" % (checkpoint_dir, model_name, step)))


if __name__ == '__main__':
    args = parse_args()
    if args.test_policy:
        test(args)
    else:
        main(args)
