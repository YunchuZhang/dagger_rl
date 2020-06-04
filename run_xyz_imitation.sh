MODE="MUJOCO_OFFLINE"
export MODE
#python -W ignore main.py --checkpoint_path=/Users/apokle/Documents/goal_conditioned_policy/dagger_rl/ckpts/headphones/save99 --mesh=mug1 --num-rollouts=30 --num-iterations=120

python -W ignore main_imitation.py --expert_data_path=/projects/katefgroup/sawyer_ddpg_weight/ --mesh=car2 --num-rollouts=1500 --num-iterations=150 \
						--checkpoint_path=/home/apokle/dagger_rl/ckpt_sawyer_baseline_imitation --mb_size=128