MODE="MUJOCO_OFFLINE"
export MODE

python gather_rollout_stats.py --expert_data_path=/projects/katefgroup/sawyer_ddpg_weight/ --num-rollouts=500