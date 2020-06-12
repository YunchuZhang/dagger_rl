# Dagger RL

This repo contains code for executing the BC distilled baseline for the quantize project.

## Setup

### Repos

The following list contains the repos to be installed. Each should be installed as a pip package or have directories included in the PATH variable.

* Quantize Baselines
    - Clone: git@github.com:gsp-27/quantize_baselines.git
    - Branch: ashwini_branch
    - Commit: c9fc525674faa45a03c659ad46c3268b822cc1f8
* Quantize Policies
    - Clone: git@github.com:ashwinipokle/quantized_policies.git
    - Branch: htung-compression
    - Commit: f1a8a4af665fb843afbdb66db259af753b264587
* Quantize Gym
    - Clone: git@github.com:gsp-27/quantize-gym.git
    - Branch: ashwini_branch
    - Commit: 25849570fe87a2dd505fb84b5c6b8d1c747bf905
* Discovery
    - Clone: git@bitbucket.org:adamharley/discovery.git
    - Branch: muojco_crop
    - Commit: 28b24cb8cd5cf75e191806beab7cdad20c8a42b4
* Multiworld
    - Clone: git@github.com:vitchyr/multiworld.git
    - Branch: ddpg_yc
    - Commit: 20efbef943b0bab4a9b6c2b1206e07fac97b2178
* Softlearning
    - Clone: git@github.com:YunchuZhang/softlearning.git
    - Branch: master
    - Commit: 6039442ae0bd0a07b29ffdc724e1908951f20e70

### Packages and Versions

Key packages and their versions are listed below.

* tensorflow-gpu==1.13.1
* mujoco-py==2.0.2.10

## Sample Command

```
python -W ignore main.py --num_rollouts=30 --num_iterations=120 --expert_data_path [path to expert checkpoint directory, whose first-level subdirectories should be mesh ids] --wandb --prefix [insert any name for the run]
```
