# Dagger RL

This repo contains code for executing the BC distilled baseline for the quantize project.

## Setup

### Repos

The following list contains the repos to be installed. Each should be installed as a pip package or have directories included in the PATH variable.

* Quantize Baselines
    - Clone: git@github.com:gsp-27/quantize_baselines.git
    - Branch: from_ashwini_branch
* Quantize Policies
    - Clone: git@github.com:ashwinipokle/quantized_policies.git
    - Branch: from_htung-compression to run this
* Quantize Gym
    - Clone: git@github.com:gsp-27/quantize-gym.git
    - Branch: from_htung_cup_branch
* Discovery
    - Clone: git@bitbucket.org:adamharley/discovery.git
    - Branch: muojco_crop
    - Commit: 28b24cb8cd5cf75e191806beab7cdad20c8a42b4

### Packages and Versions

Key packages and their versions are listed below.

* tensorflow-gpu==1.13.1
* mujoco-py==2.0.2.10

## Sample Command

```
python -W ignore main.py --num_rollouts=30 --num_iterations=120 --expert_data_path [path to expert checkpoint directory, whose first-level subdirectories should be mesh ids] --wandb --prefix [insert any name for the run]
```
