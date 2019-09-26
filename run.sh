MODE="MUJOCO_OFFLINE"
export MODE
python -W ignore main_3d.py --checkpoint_path=/projects/katefgroup/yunchu/mug148/checkpoint_1400/ --mesh=mug1 --num-rollouts=30 --num-iterations=120