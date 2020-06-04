MODE="MUJOCO_OFFLINE"
export MODE

python policy_compression_multiscale.py --expert_data_path=/Users/apokle/Documents/quantized_policies/trained_models_fetch/fetch_cup/159e56c18906830278d8f8c02c47cde0/models/  \
		--expert_log_path=/Users/apokle/Documents/quantized_policies/trained_models_fetch/fetch_cup/159e56c18906830278d8f8c02c47cde0/logs \
		--num-rollouts=25