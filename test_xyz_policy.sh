MODE="MUJOCO_OFFLINE"
export MODE
python -W ignore main.py --checkpoint_path=/home/apokle/dagger_rl/ckpt_sawyer_baseline_dagger/baseline_3obj_no_staircase_lr_0.005_steps_20000_decay_0.96_shuffle --mesh=mug1 --test-num-rollouts=1000 --test-policy=True


####### car2 Good results - trained and tested on 1 obj - car2 - goal reach 70-80% - abs value in observation
#python -W ignore main.py --checkpoint_path=/home/apokle/dagger_rl/ckpt_sawyer_baseline_dagger/baseline_1obj_no_staircase_lr_0.005_steps_20000_decay_0.96_shuffle --mesh=car2 --test-num-rollouts=1000 --test-policy=True

