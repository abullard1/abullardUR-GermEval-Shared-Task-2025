# Competition-Solution/src/wandb_utils.py

"""
Weights and Biases Integration Utilities

This module provides experiment tracking using Weights & Biases. 
It includes functions for authentication, run initialization, 
artifact management, and sweep configuration.

Author: Samuel Ruairí Bullard
Project: Evaluating Cost-Sensitive Loss Functions for Transformer-Based German Harmful Content Detection
Date: January 2026
"""

import wandb
import dotenv
import os

# Logs in to W&B
def login_wandb(dotenv_path=None):
    # Finds the .env file
    if dotenv_path is None:
        env_path = dotenv.find_dotenv()
    else:
        env_path = dotenv_path

    # Gets the API key
    api_key = dotenv.get_key(env_path, "WANDB_API_KEY")
    if not api_key:
        print("WANDB_API_KEY not found in .env file. Disabling W&B reporting.")
        return False

    # Attempts to login
    try:
        success = wandb.login(key=api_key, timeout=10)
        if not success:
            print("W&B login failed. Disabling W&B reporting.")
            return False
        print("Successfully logged in to W&B")
        return True
    except Exception as e:
        print(f"Error logging in to W&B: {e}. Disabling W&B reporting.")
        return False


# Initializes the W&B run
def init_wandb(loaded_config, dataset_mode, logging_dir, version_tag=None):
    os.environ['WANDB_DIR'] = str(logging_dir)
    # Extracts the main config sections
    wandb_cfg = loaded_config.get("wandb", {})
    
    # Builds the run name: {dataset_mode}-{subtask_id}-{model_arch}-{experiment_type}-{version}
    run_name_parts = [
        loaded_config["dataset_modes"][dataset_mode]["suffix"],
        loaded_config["subtask_id"],
        loaded_config["wandb_run_name_parts"]["model_arch"],
        loaded_config["experiment_type"]
    ]
    if version_tag:
        run_name_parts.append(version_tag)
    run_name = "-".join(run_name_parts)

    # Builds the group name: {dataset_mode}/{subtask_name}/{model_family}
    group_name = "/".join([
        loaded_config["dataset_modes"][dataset_mode]["suffix"],
        loaded_config["subtask_name"],
        loaded_config["model_family"]
    ])

    # Builds the tags
    tags = [
        loaded_config["dataset_modes"][dataset_mode]["suffix"],
        loaded_config["subtask_name"],
        loaded_config["model_family"],
        loaded_config["wandb_run_name_parts"]["model_arch"],
        loaded_config["experiment_type"]
    ]
    if version_tag:
        tags.append(version_tag)
    
    # Adds custom tags from config
    custom_tags = loaded_config.get("wandb_tags_add", [])
    if custom_tags:
        tags.extend(custom_tags)
    tags = list(set(tags))  # Remove duplicates

    # Prepares the config for W&B logging
    wandb_config = {
        "source_config": loaded_config,
        "dataset_mode": dataset_mode,
        "version_tag": version_tag,
        "run_name": run_name,
        "group_name": group_name,
        "job_type": "training"
    }

    # Initializes the W&B run
    # https://docs.wandb.ai/ref/python/init/
    run = wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        name=run_name,
        config=wandb_config,
        tags=tags,
        group=group_name,
        notes=loaded_config.get("wandb_notes", f"Run for {run_name}"),
        job_type="training",
        reinit=True,
        save_code=wandb_cfg.get("save_code"),
        sync_tensorboard=wandb_cfg.get("sync_tensorboard"),
        dir=logging_dir
    )

    print(f"Initialized W&B run: {run.name} (Group: {run.group}, ID: {run.id})")
    return run


# Saves and uploads the model to W&B as an artifact
def save_and_upload_model_to_wandb(run, name, model_type, description, metadata, local_path, target_path):
    try:
        # Creates and saves the artifact
        artifact = wandb.Artifact(
            name=name,
            type=model_type,
            description=description
        )
        
        # Adds the model directory
        artifact.add_dir(local_path)
        
        # Adds metadata
        artifact.metadata.update(metadata)
        
        # Logs the artifact to the run
        run.log_artifact(artifact)
        
        # Links the artifact to the target path
        run.link_artifact(artifact=artifact, target_path=target_path)
        
        print(f"Successfully created and uploaded W&B artifact: {artifact.name}")
        return artifact
    except Exception as e:
        print(f"Error when saving and uploading the artifact to W&B: {e}")
        raise



# Creates temporary config with W&B sweep parameters during hyperparameter optimization
# This DOES NOT modify our original YAML files - only creates in-memory configs for each sweep run
# After a sweeo us done we manually copy and paste the best hyperparameters into the config to be used for the submission model training
def update_config_with_sweep_params(config):
    # Creates a copy to avoid modifying the original config
    updated_config = config.copy()
    
    # Gets all sweep parameters that W&B is testing in this run
    sweep_params = dict(wandb.config)
    
    # Converts sweep parameters to the correct types (W&B passes everything as strings)
    for key, value in sweep_params.items():
        if key in ['learning_rate', 'weight_decay', 'warmup_ratio', 'gamma']:
            sweep_params[key] = float(value)
        elif key in ['num_train_epochs', 'per_device_train_batch_size', 'per_device_eval_batch_size', 
                     'warmup_steps', 'save_steps', 'eval_steps', 'logging_steps']:
            sweep_params[key] = int(value)
    
    # Gamma is only relevant to our focal loss approaches, not HuggingFace TrainingArguments
    if 'gamma' in sweep_params:
        updated_config['gamma'] = sweep_params['gamma']
        sweep_params.pop('gamma')
    
    # All remaining parameters go to training_arguments (learning_rate, epochs, etc.)
    updated_config['training_arguments'].update(sweep_params)
    
    return updated_config


# Logs evaluation metrics to an existing W&B run by name
# We use this during the evaluation of a single model in the single_model_evaluation_utils module
def log_evaluation_metrics_to_run(run_name, metrics, entity, project, prefix="test_"):
    try:
        if not login_wandb():
            return False

        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")

        for run in runs:
            if run.name != run_name:
                continue

            wandb_metrics = {}

            # Loops through the metrics and logs them to the W&B run
            for key, value in metrics.items():
                if key == "per_class_metrics":
                    for class_name, class_metrics in value.items():
                        clean_name = class_name.replace('class_', '')
                        for metric_name, metric_value in class_metrics.items():
                            wandb_metrics[f"{prefix}{metric_name}_class_{clean_name}"] = float(metric_value)
                else:
                    wandb_metrics[f"{prefix}{key}"] = value

            # Updates the summary and logs the metrics to the W&B run
            run.summary.update(wandb_metrics)
            run.log(wandb_metrics)
            print(f"Logged evaluation metrics to W&B run: {run_name}")
            return True

        print(f"No W&B run found with name: {run_name}")
        return False
    except Exception as e:
        print(f"W&B logging error: {e}")
        return False


# Analyzes sweep results and finds the best run
def get_sweep_best_run(sweep_id, project):
    api = wandb.Api()
    sweep = api.sweep(f"{project}/{sweep_id}")
    runs = list(sweep.runs)
    
    if not runs:
        print("No runs found in sweep")
        return None, None
    
    # Finds the best run by target metric (This will always be macro-F1 in our case)
    metric_name = sweep.config['metric']['name']
    goal = sweep.config['metric']['goal']
    
    # W&B logs metrics with '/' but configs use '_' - normalize to W&B format
    metric_name = metric_name.replace('_', '/')
    
    best_run = None
    best_value = float('-inf') if goal == 'maximize' else float('inf')
    
    # Loops through the runs and finds the best run
    for run in runs:
        if run.state == 'finished' and metric_name in run.summary:
            value = run.summary[metric_name]
            if (goal == 'maximize' and value > best_value) or (goal == 'minimize' and value < best_value):
                best_value = value
                best_run = run
    
    return best_run, best_value

# Prints best hyperparameters in YAML format to copy and paste them into the config to be used for the submission model training
def print_best_hyperparams(sweep_id, project):
    best_run, best_value = get_sweep_best_run(sweep_id, project)
    
    if not best_run:
        print("No best run found")
        return
    
    # Gets the metric name for display
    api = wandb.Api()
    sweep = api.sweep(f"{project}/{sweep_id}")
    metric_name = sweep.config['metric']['name'].replace('_', '/')
    
    print(f"\n🎯 BEST RUN: {best_run.name}")
    print(f"📊 Best {metric_name}: {best_value:.5f}")
    print(f"🔗 Run URL: {best_run.url}")
    
    print(f"\n📋 COPY-PASTE HYPERPARAMETERS:")
    
    # Extracts key hyperparameters for the config training arguments
    print("training_arguments:")
    training_params = ['learning_rate', 'num_train_epochs', 'per_device_train_batch_size', 
                      'per_device_eval_batch_size', 'warmup_steps', 'weight_decay']
    
    # Loops through the training parameters and prints them
    for param in training_params:
        if param in best_run.config:
            value = best_run.config[param]
            print(f"  {param}: {value}")
    
    # Extracts gamma if present (Only relevant to our focal loss approaches)
    if 'gamma' in best_run.config:
        print(f"\n# Focal Loss Configuration")
        print(f"gamma: {best_run.config['gamma']}")


# Creates and runs a W&B sweep to tune the hyperparameters of a model
# Check the respective config yamls for the sweep configuration ranges
# that are valid for the sweep
def create_and_run_sweep(config, count=10, train_with_sweeps=None):
    # Logs in to W&B
    if not login_wandb():
        raise Exception("W&B login required for sweeps")
    
    sweep_config = config['wandb_sweep_config']
    project = config['wandb']['project']
    
    # Creates the sweep
    sweep_id = wandb.sweep(sweep_config, project=project)
    
    # Runs the sweep agent
    wandb.agent(sweep_id, function=train_with_sweeps, count=count)
    
    # Analyzes and prints best results using the function we defined above
    print_best_hyperparams(sweep_id, project)
    
    return sweep_id
