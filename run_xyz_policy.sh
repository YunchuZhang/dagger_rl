MODE="MUJOCO_OFFLINE"
export MODE
python -W ignore main.py --checkpoint_path=/Users/apokle/Documents/goal_conditioned_policy/dagger_rl/ckpts/headphones/save99 --mesh=mug1 --num-rollouts=30 --num-iterations=120