MODE="MUJOCO_OFFLINE"
export MODE
export GPV_SOURCE_DIR='/home/yunchuz/fetchtemp/quantized_policies/quantized_policies'
# python -W ignore main.py --num_rollouts=30 --num_iterations=120 --expert_data_path='/projects/katefgroup/quantized_policies/checkpoints/trained_models/2020-06-10_01-32-01/ckpts/' --wandb 
# --prefix='yc_test'
python -W ignore main_3d.py --num_rollouts=30 --num_iterations=120 --expert_data_path='/projects/katefgroup/quantized_policies/checkpoints/trained_models/2020-06-10_01-32-01/ckpts/' --wandb 
--prefix 3d_end2end
