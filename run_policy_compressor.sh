MODE="MUJOCO_OFFLINE"
export MODE

python policy_compression.py --expert_data_path=/projects/katefgroup/sawyer_ddpg_weight/push --num-rollouts=500