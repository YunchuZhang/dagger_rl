MODE="MUJOCO_OFFLINE"
export MODE
python -W ignore main_3d.py --mesh=mug1 --num-rollouts=30 --num-iterations=120