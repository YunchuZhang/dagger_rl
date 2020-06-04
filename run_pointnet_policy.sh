MODE="MUJOCO_OFFLINE"
export MODE

python -W ignore main_pointnet.py --expert_data_path=/projects/katefgroup/sawyer_ddpg_weight/ --mesh=mug1 --num-rollouts=30 --num-dagger-rollouts=20 --num-iterations=150 \
						--checkpoint_path=/home/apokle/dagger_rl/ckpt_sawyer_baseline_dagger --mb_size=128