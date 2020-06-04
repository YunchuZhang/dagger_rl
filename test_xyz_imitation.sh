MODE="MUJOCO_OFFLINE"
export MODE

#python -W ignore main_imitation.py --checkpoint_path=/home/apokle/dagger_rl/ckpt_sawyer_baseline_imitation/baseline_3obj_imitation_shuffle \
#			--mesh=car2 --test-num-rollouts=100 --test-policy=True

python -W ignore main_imitation.py --checkpoint_path=/home/apokle/dagger_rl/ckpt_sawyer_baseline_imitation/baseline_3obj_imitation_shuffle_1500 \
			--mesh=car2 --test-num-rollouts=100 --test-policy=True
