import os
import json
import pickle
from softlearning.environments.utils import get_environment_from_params, get_environment_from_params_custom
from softlearning.policies.utils import get_policy_from_variant

def get_policy(checkpoint_path):
    checkpoint_path = checkpoint_path.rstrip('/')
    experiment_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(experiment_path, 'params.json')
    with open(variant_path, 'r') as f:
        variant = json.load(f)

    environment_params = (
        variant['environment_params']['evaluation']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])
    evaluation_environment = get_environment_from_params(environment_params)

    policy = (get_policy_from_variant(variant, evaluation_environment, Qs=[None]))
    training_environment = get_environment_from_params_custom(environment_params)

    return policy, training_environment