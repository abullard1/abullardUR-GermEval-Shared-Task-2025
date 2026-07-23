# Competition-Solution/src/single_model_evaluation_utils.py

"""
Single Model Evaluation Utilities

This script rovides utilities for evaluating a single fine-tuned model that we trained
and submitted to the GermEval 2025 Shared Task/Competition. It includes functions for measuring the 
following metrics:
- Macro-F1 Score
- Accuracy
- Macro Precision & Recall
- Per-class F1, Precision, Recall
- Matthews Correlation Coefficient
- Bootstrap confidence intervals
- ROC-AUC (binary subtasks)
- Average Precision (binary subtasks)
- Error analysis summary
- Confusion matrix

Author: Samuel Ruairí Bullard
Project: Evaluating Cost-Sensitive Loss Functions for Transformer-Based German Harmful Content Detection
Date: January 2026
"""


import warnings
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
import torch
from transformers import AutoProcessor, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from pathlib import Path
from datasets import load_from_disk
import ipywidgets as widgets
from IPython.display import clear_output, display
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)
from scipy.stats import bootstrap
import evaluate

# Local imports
import config_utils, wandb_utils
from plot_theme import apply_plot_theme
from visualizations_single import create_visualizations

warnings.filterwarnings("ignore")
console = Console()

# External dataset evaluation - GermEval 2018
# Gets the path via relative path to the project root
def get_germeval2018_path(project_root):
    return str(Path(project_root) / "data" / "evaluation" / "germeval2018" / "germeval2018.test.txt")

# Gets the project root
project_root_dir = Path.cwd().parent.parent.parent  
# Loads the base config yaml file
base_config = config_utils.load_yaml(str(project_root_dir / "configs" / "base.yaml"))

# Competition baselines (from the competition organizers)
BASELINES = {"c2a": {"f1": 0.59}, "dbo": {"f1": 0.47}, "vio": {"f1": 0.69}}

# Class names/Labels for the subtasks
CLASS_NAMES = {
    "c2a": ["False", "True"],
    "dbo": ["Nothing", "Criticism", "Agitation", "Subversive"], 
    "vio": ["False", "True"]
}


# Discovers the models within the models directory
def discover_models(project_root):
    models_dir = Path(project_root) / "models" / "finetuned_models"
    discovered_models = {}
    
    # Iterates through the subtasks
    for subtask in ["c2a", "dbo", "vio"]:
        subtask_dir = models_dir / subtask
        if subtask_dir.exists():
            subtask_models = []
            for model_dir in subtask_dir.iterdir():
                if model_dir.is_dir():
                    subtask_models.append({"name": model_dir.name, "path": model_dir})
            discovered_models[subtask] = subtask_models
    
    return discovered_models


# Parses a model directory name into its components (dataset, subtask, model, experiment)
def parse_model_name(model_name):
    parts = model_name.split("-")

    # Validates that we have the correct number of parts
    if len(parts) < 4:
        return None
    
    # Derives the experiment type from the tokens
    experiment_type = None

    # Checks if the experiment type is in the tokens
    for token in parts:
        if token in {"best", "train", "baseline"}:
            experiment_type = token
            break

    if experiment_type is None:
        experiment_type = parts[3]
    
    # Returns the parsed model name
    return {
        "dataset_mode": parts[0],
        "subtask_id": parts[1],
        "model_arch": parts[2],
        "experiment_type": experiment_type,
    }


# Builds the evaluation configuration for a model name using predictable dataset paths
def get_eval_config(model_name, project_root):
    parsed = parse_model_name(model_name)

    # Validates that the model name is valid
    if not parsed:
        raise ValueError(f"Invalid model name: {model_name}")

    # Derives the subtask ID and dataset mode from the parsed model name
    subtask = parsed["subtask_id"]
    if parsed["dataset_mode"] == "train":
        dataset_suffix = f"{subtask}/{subtask}_hf_dataset_tokenized_train"
    else:
        dataset_suffix = f"{subtask}/{subtask}_hf_dataset_tokenized"

    # Derives the final dataset path
    final_dataset_path = str(Path(project_root) / "data" / "processed" / dataset_suffix)

    # Anonymization special tokens
    default_special_tokens = ["[@URL]", "[@EMAIL]", "[@PRE]", "[@POL]", "[@GRP]", "[@IND]"]

    # Returns the evaluation configuration
    return {
        "subtask_id": subtask,
        "model_path": str(Path(project_root) / "models" / "finetuned_models" / subtask / model_name / "best_model_eval_f1-macro"),
        "dataset_path": final_dataset_path,
        "special_tokens": default_special_tokens,
        "tokenization": {"special_tokens": default_special_tokens},
        "is_binary": subtask in ["c2a", "vio"]
    }

# Loads the model, processor, and evaluation split based on the provided configuration
def load_model_and_data(config):
    # Defines the device to use for the model (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loads the processor and adds the special tokens
    processor = AutoProcessor.from_pretrained(config["model_path"], trust_remote_code=True)
    processor.add_special_tokens({"additional_special_tokens": config["special_tokens"]})
    
    # Loads the model with half precision if CUDA is available, otherwise uses full precision (CPU)
    if torch.cuda.is_available():
        model = AutoModelForSequenceClassification.from_pretrained(config["model_path"], torch_dtype=torch.float16)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(config["model_path"])

    # Resizes the token embeddings to the length of the processor
    model.resize_token_embeddings(len(processor), mean_resizing=True)
    model.to(device).eval()
    
    # Loads the pre-tokenized dataset from the same path used for training/evaluation (i.e. the validation set)
    dataset = load_from_disk(config["dataset_path"])
    eval_split = dataset["validation"]
    console.print(f"[green][OK] Loaded model on {device}, validation samples: {len(eval_split)}[/green]")
    return model, processor, eval_split

# Runs batched prediction and returns predictions, probabilities, and true labels
def predict(model, processor, dataset, batch_size=16):
    string_cols = []

    # Removes string columns (e.g. "id", "description") before batching, as
    # DataCollatorWithPadding cannot convert them to tensors
    for col_name, feature in dataset.features.items():
        if feature.dtype == "string":
            string_cols.append(col_name)
    if string_cols:
        dataset = dataset.remove_columns(string_cols)

    trainer = Trainer(
        model=model, 
        args=TrainingArguments(
            output_dir="./tmp_eval", 
            per_device_eval_batch_size=batch_size,
            dataloader_drop_last=False, 
            report_to="none",
            disable_tqdm=True,
            label_names=["labels"] 
        ),
        data_collator=DataCollatorWithPadding(processor)
    )

    # Runs the predictions
    predictions = trainer.predict(dataset, ignore_keys=["id", "description"])
    logits = predictions.predictions
    true_labels = predictions.label_ids

    # Converts the logits to probabilities and gets the final predictions
    probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()
    final_predictions = np.argmax(logits, axis=1)
    
    return final_predictions, probabilities, true_labels

# Computes core metrics and per-class metrics via HF evaluate
def _compute_metrics_with_evaluate(true_labels, predicted_labels, predicted_probabilities=None, is_binary=False):
    metrics = {}

    # Core metrics
    metrics["accuracy"] = float(evaluate.load("accuracy").compute(predictions=predicted_labels, references=true_labels)["accuracy"])
    metrics["macro_f1"] = float(evaluate.load("f1").compute(predictions=predicted_labels, references=true_labels, average="macro")["f1"])
    metrics["macro_precision"] = float(evaluate.load("precision").compute(predictions=predicted_labels, references=true_labels, average="macro")["precision"])
    metrics["macro_recall"] = float(evaluate.load("recall").compute(predictions=predicted_labels, references=true_labels, average="macro")["recall"])
    metrics["matthews_corrcoef"] = float(evaluate.load("matthews_correlation").compute(predictions=predicted_labels, references=true_labels)["matthews_correlation"])

    # Per-class metrics via sklearn
    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, zero_division=0
    )

    # Creates a dictionary of the per-class metrics
    per_class_metrics = {}
    for i in range(len(precision_arr)):
        per_class_metrics[f"class_{i}"] = {
            "precision": float(precision_arr[i]),
            "recall": float(recall_arr[i]),
            "f1_score": float(f1_arr[i]),
            "support": int(support_arr[i])
        }

    # Adds the per-class metrics to the metrics dictionary
    metrics["per_class_metrics"] = per_class_metrics

    # Binary-specific metrics via HF evaluate again
    # Checks if the subtask is binary and if the probabilities are not None
    if is_binary and predicted_probabilities is not None:
        if not (predicted_probabilities.ndim == 2 and predicted_probabilities.shape[1] == 2):
            raise ValueError("Expected binary probability array of shape (N, 2) for is_binary=True.")

        # Gets the positive class scores
        positive_class_scores = predicted_probabilities[:, 1]

        # Computes the ROC-AUC and average precision
        metrics["roc_auc"] = float(evaluate.load("roc_auc").compute(references=true_labels, prediction_scores=positive_class_scores)["roc_auc"])
        metrics["avg_precision"] = float(average_precision_score(true_labels, positive_class_scores))

    return metrics

# Computes bootstrap confidence intervals for the selected metrics
# ROC-AUC for binary problems
def compute_confidence_intervals(true_labels, predicted_labels, predicted_probabilities=None, is_binary=None, n_bootstraps=1000, confidence_level=0.95):
    bootstrap_data = (
        (true_labels, predicted_labels, predicted_probabilities)
        if predicted_probabilities is not None
        else (true_labels, predicted_labels)
    )
    
    # Defines the bootstrap statistic functions
    def accuracy_stat(sample_true_labels, sample_predicted_labels, *args):
        return accuracy_score(sample_true_labels, sample_predicted_labels)
    
    def f1_stat(sample_true_labels, sample_predicted_labels, *args):
        return f1_score(sample_true_labels, sample_predicted_labels, average="macro")
    
    def precision_stat(sample_true_labels, sample_predicted_labels, *args):
        return precision_score(sample_true_labels, sample_predicted_labels, average="macro", zero_division=0)
    
    def recall_stat(sample_true_labels, sample_predicted_labels, *args):
        return recall_score(sample_true_labels, sample_predicted_labels, average="macro", zero_division=0)
    
    def mcc_stat(sample_true_labels, sample_predicted_labels, *args):
        return matthews_corrcoef(sample_true_labels, sample_predicted_labels)
    
    # Creates a dictionary of the metric functions
    metric_functions = {
        "accuracy": accuracy_stat,
        "macro_f1": f1_stat,
        "macro_precision": precision_stat,
        "macro_recall": recall_stat,
        "matthews_corrcoef": mcc_stat
    }
    
    # Creates a dictionary of the confidence intervals
    confidence_intervals = {}

    # Loops through the metric functions and computes the confidence intervals
    for metric_name, metric_func in metric_functions.items():
        try:
            res = bootstrap(bootstrap_data, metric_func, n_resamples=n_bootstraps,
                          confidence_level=confidence_level, random_state=42, paired=True, method="BCa")
            confidence_intervals[metric_name] = {
                "lower": float(res.confidence_interval.low),
                "upper": float(res.confidence_interval.high)
            }
        except Exception:
            confidence_intervals[metric_name] = {"lower": np.nan, "upper": np.nan}
    
    # Computes the binary ROC-AUC if applicable
    if (is_binary is True) and (predicted_probabilities is not None):
        def roc_stat(sample_true_labels, sample_predicted_labels, sample_predicted_probabilities):
            positive_class_probabilities = sample_predicted_probabilities[:, 1]
            try:
                return roc_auc_score(sample_true_labels, positive_class_probabilities)
            except:
                return np.nan
        
        # Computes the ROC-AUC confidence interval
        try:
            res = bootstrap(bootstrap_data, roc_stat, n_resamples=n_bootstraps,
                          confidence_level=confidence_level, random_state=42, paired=True, method="BCa")
            confidence_intervals["roc_auc"] = {
                "lower": float(res.confidence_interval.low),
                "upper": float(res.confidence_interval.high)
            }
        except Exception:
            confidence_intervals["roc_auc"] = {"lower": np.nan, "upper": np.nan}
    
    return confidence_intervals


# Generates an error analysis dataframe for misclassified samples
def generate_error_dataframe(true_labels, predicted_labels, predicted_probabilities=None, texts=None):
    # Gets the indices of the misclassified samples (i.e. where the true label is not equal to the predicted label)
    misclassified_indices = np.where(true_labels != predicted_labels)[0]

    # If there are no misclassified samples, returns an empty dataframe
    if len(misclassified_indices) == 0:
        return pd.DataFrame()
    
    # Creates a dictionary of the error data (index, true label, predicted label)
    error_data = {
        "index": misclassified_indices,
        "true_label": true_labels[misclassified_indices],
        "predicted_label": predicted_labels[misclassified_indices]
    }
    
    # If the probabilities are not None, adds the confidence scores and predicted probabilities to the error data
    if predicted_probabilities is not None:
        if predicted_probabilities.ndim > 1:
            confidence_scores = np.max(predicted_probabilities[misclassified_indices], axis=1)
            predicted_class_probability = predicted_probabilities[misclassified_indices, predicted_labels[misclassified_indices]]
        else:
            confidence_scores = np.abs(predicted_probabilities[misclassified_indices] - 0.5) + 0.5
            predicted_class_probability = predicted_probabilities[misclassified_indices]
        
        error_data.update({
            "confidence": confidence_scores,
            "predicted_proba": predicted_class_probability
        })
    
    # If the texts are not None, adds the misclassified texts to the error data
    if texts is not None:
        # Adds the misclassified texts corresponding to the indices
        collected_texts = []
        for i in misclassified_indices:
            collected_texts.append(texts[i])
        error_data["text"] = collected_texts
    
    # Creates a dataframe of the error data
    error_df = pd.DataFrame(error_data)
    
    # Sorts the dataframe by the confidence scores
    if "confidence" in error_df.columns:
        error_df = error_df.sort_values("confidence", ascending=False)
    
    return error_df


# Orchestrates the full metric computation with optional CI and error analysis
def compute_metrics(true_labels, predicted_labels, predicted_probabilities=None, subtask_id=None, class_names=None, 
                   include_confidence_intervals=False, include_error_analysis=False, texts=None):
    # Determines the binary classification statically from subtask when available

    # Checks if the subtask is binary
    if subtask_id in {"c2a", "vio"}:
        is_binary = True
    elif subtask_id == "dbo":
        is_binary = False
    else:
        is_binary = (len(np.unique(true_labels)) == 2)

    # Computes the metrics
    metrics = _compute_metrics_with_evaluate(true_labels, predicted_labels, predicted_probabilities, is_binary=is_binary)

    # Computes the confidence intervals if requested
    if include_confidence_intervals:
        metrics["confidence_intervals"] = compute_confidence_intervals(true_labels, predicted_labels, predicted_probabilities, is_binary=is_binary)
    
    # Computes the error analysis if requested
    if include_error_analysis:
        error_df = generate_error_dataframe(true_labels, predicted_labels, predicted_probabilities, texts)
        if not error_df.empty:
            metrics["error_analysis"] = error_df
    
    # Computes the baseline F1 score and improvement
    if subtask_id in BASELINES:
        baseline = BASELINES[subtask_id]["f1"]
        improvement = metrics["macro_f1"] - baseline

        # Updates the metrics dictionary with the baseline F1 score and improvement
        metrics.update({
            "baseline_f1": baseline,
            "improvement": float(improvement),
            "beats_baseline": improvement > 0
        })
    
    return metrics

# Evaluates the transfer capability of our models on the GermEval-2018 dataset
# https://github.com/uds-lsv/GermEval-2018-Data
def evaluate_on_germeval2018(model, processor, project_root, sample_size=None):
    from datasets import load_dataset
    
    # Gets the path to the GermEval-2018 dataset
    dataset_path = get_germeval2018_path(project_root)
    if not Path(dataset_path).exists():
        console.print(f"[red]GermEval 2018 dataset not found at: {dataset_path}")
        return None

    # Loads the dataset from the path
    dataset = load_dataset("csv", data_files={"test": dataset_path})["test"]

    # Samples the dataset if needed 
    if sample_size and len(dataset) > sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))

    # Function to tokenize a batch of examples
    # 1024 tokens max length, just as we had in the training scripts
    max_len = 1024

    def tokenize_batch(examples):
        return processor(
            examples["text"],
            padding="max_length", 
            truncation=True,
            max_length=max_len
        )

    # Tokenizes the dataset
    tokenized_dataset = dataset.map(tokenize_batch, batched=True)

    # Predicts and evaluates the model
    predictions, probabilities, true_labels = predict(model, processor, tokenized_dataset, batch_size=32)
    
    # Computes the metrics via the above defined function
    metrics = _compute_metrics_with_evaluate(true_labels, predictions, probabilities, is_binary=True)
    metrics.update({"dataset_name": "germeval2018", "sample_size": len(dataset)})
    
    console.print(f"[cyan]GermEval 2018 evaluation: Macro-F1 = {metrics['macro_f1']:.3f}")
    return metrics

# Logs the metrics to an existing W&B run from the training run
def log_to_wandb(run_name, metrics):
    # Gets the entity and project from the base config
    try:
        wandb_cfg = base_config.get("wandb", {})
        entity = wandb_cfg.get("entity")
        project = wandb_cfg.get("project")

        # Logs the metrics to the W&B run
        ok = wandb_utils.log_evaluation_metrics_to_run(run_name, metrics, entity, project, prefix="test_")
        if ok:
            console.print(f"[green][OK] Logged to W&B: {run_name}[/green]")
        else:
            console.print(f"[yellow]No W&B run found or logging skipped: {run_name}[/yellow]")
        return ok
    
    except Exception as e:
        console.print(f"[red]W&B error: {e}[/red]")
        return False

# Runs the full evaluation pipeline for a model and returns metrics and predictions
def evaluate_model(model_name, project_root, update_wandb=True, 
                  include_confidence_intervals=False, include_error_analysis=False, 
                  include_external_eval=False):
    console.print(f"[bold green]Evaluating: {model_name}[/bold green]")
    
    config = get_eval_config(model_name, project_root)
    model, processor, dataset = load_model_and_data(config)

    # Main validation evaluation
    predictions, probabilities, true_labels = predict(model, processor, dataset)

    metrics = compute_metrics(
        true_labels, predictions, probabilities, 
        config["subtask_id"], CLASS_NAMES[config["subtask_id"]],
        include_confidence_intervals=include_confidence_intervals,
        include_error_analysis=include_error_analysis,
        texts=dataset["description"] if "description" in dataset.column_names else None
    )
    
    # External evaluation for transfer capability (only for the binary tasks)
    external_results = {}
    if include_external_eval and config["is_binary"]:
        console.print("[yellow]Running external evaluation (GermEval-2018 coarse)...[/yellow]")
        ext_metrics = evaluate_on_germeval2018(model, processor, project_root, sample_size=None)
        if ext_metrics:
            external_results["germeval2018"] = ext_metrics
    

    if external_results:
        metrics["external_evaluation"] = external_results
    
    if update_wandb:
        wandb_metrics = {}
        for key, value in metrics.items():
            # Filters out bulky nested objects before logging to W&B
            if key in ["confidence_intervals", "error_analysis", "external_evaluation"]:
                continue
            wandb_metrics[key] = value

        # Logs the metrics to the W&B run
        log_to_wandb(model_name, wandb_metrics)
    
    # Clears the model and processor from the cache
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "config": config, 
        "metrics": metrics, 
        "predictions": predictions, 
        "probabilities": probabilities, 
        "true_labels": true_labels
    }

# --- Reporting & Visualization ---

# Prints the main metrics table
def print_main_metrics(metrics, subtask_id):
    table = Table(title=f"{subtask_id.upper()} - Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Adds the confidence interval column if available
    has_confidence_intervals = "confidence_intervals" in metrics
    if has_confidence_intervals:
        table.add_column("95% CI", style="yellow")
    
    # Helper function to format the confidence intervals
    def format_confidence_interval(metric_name):
        if has_confidence_intervals and metric_name in metrics["confidence_intervals"]:
            confidence_interval = metrics["confidence_intervals"][metric_name]
            if not (np.isnan(confidence_interval["lower"]) or np.isnan(confidence_interval["upper"])):
                return f"[{confidence_interval['lower']:.4f}, {confidence_interval['upper']:.4f}]"
        return ""
    
    # Core metrics
    main_metrics_to_show = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "matthews_corrcoef"]
    for metric_name in main_metrics_to_show:
        row = [metric_name.replace("_", " ").title(), f"{metrics[metric_name]:.4f}"]
        if has_confidence_intervals:
            row.append(format_confidence_interval(metric_name))
        table.add_row(*row)
    
    # Baseline comparison
    if "baseline_f1" in metrics:
        row = ["Baseline F1", f"{metrics['baseline_f1']:.4f}"]
        if has_confidence_intervals: row.append("")
        table.add_row(*row)
        
        improvement = metrics["improvement"]
        improvement_status = "UP" if improvement > 0 else "DOWN"
        row = ["Improvement", f"{improvement_status} {improvement:+.4f}"]
        if has_confidence_intervals: row.append("")
        table.add_row(*row)
    
    # Binary-specific metrics
    if "roc_auc" in metrics:
        row = ["ROC AUC", f"{metrics['roc_auc']:.4f}"]
        if has_confidence_intervals:
            row.append(format_confidence_interval("roc_auc"))
        table.add_row(*row)
        
    if "avg_precision" in metrics:
        row = ["Average Precision", f"{metrics['avg_precision']:.4f}"]
        if has_confidence_intervals: row.append("")
        table.add_row(*row)
    
    console.print(table)

# Prints the per-class metrics table
def print_per_class_metrics(metrics):
    if "per_class_metrics" not in metrics:
        return
        
    console.print("\n[bold cyan]Per-Class Metrics:[/bold cyan]")
    table = Table()
    table.add_column("Class", style="cyan")
    table.add_column("Precision", style="green")
    table.add_column("Recall", style="green")
    table.add_column("F1-Score", style="green")
    table.add_column("Support", style="yellow")
    
    for class_name, class_metrics in metrics["per_class_metrics"].items():
        table.add_row(
            class_name.replace("class_", ""),
            f"{class_metrics['precision']:.4f}",
            f"{class_metrics['recall']:.4f}",
            f"{class_metrics['f1_score']:.4f}",
            str(int(class_metrics["support"]))
        )
    
    console.print(table)

# Prints a full summary of the evaluation results
def print_results(results):
    metrics = results["metrics"]
    config = results["config"]
    
    print_main_metrics(metrics, config["subtask_id"])
    print_per_class_metrics(metrics)


# Handles the evaluation click event
def _handle_evaluation_click(model_name, update_wandb, include_external, project_root, output_area):
    with output_area:
        clear_output(wait=True)
        try:
            # Evaluates the model
            results = evaluate_model(
                model_name=model_name,
                project_root=project_root,
                update_wandb=update_wandb,
                include_confidence_intervals=True,
                include_error_analysis=True,
                include_external_eval=include_external
            )
            
            # Prints the results
            print_results(results)
            create_visualizations(
                results,
                CLASS_NAMES[results["config"]["subtask_id"]]
            )
            
            # Displays the external evaluation results
            if "external_evaluation" in results["metrics"]:
                console.print("\n[bold cyan]External Evaluation (Transfer Capability):[/bold cyan]")
                for dataset_name, ext_metrics in results["metrics"]["external_evaluation"].items():
                    console.print(f"  {dataset_name}: Macro-F1 = {ext_metrics['macro_f1']:.3f}, "
                                f"Accuracy = {ext_metrics['accuracy']:.3f} (n={ext_metrics['sample_size']})")
            
            # Displays the error analysis
            if "error_analysis" in results["metrics"]:
                console.print("\n[bold red]Error Analysis (Top 10 Most Confident Errors):[/bold red]")
                display(results["metrics"]["error_analysis"].head(10))
            else:
                console.print("\n[bold green][OK] No misclassifications found.[/bold green]")

        except Exception as e:
            console.print(f"[bold red]An error occurred during evaluation:[/bold red]\n{e}")
            import traceback
            traceback.print_exc()

# Creates the evaluation widget
def create_evaluation_widget(project_root):
    apply_plot_theme()
    available_models = discover_models(project_root)
    
    # Creates the model dropdown
    all_models = []
    for subtask, models in available_models.items():
        for model in models:
            display_name = f"{subtask.upper()} | {model['name']}"
            all_models.append((display_name, model["name"]))
    
    # Creates the model dropdown
    model_dropdown = widgets.Dropdown(
        options=all_models,
        description="Model:",
        layout=widgets.Layout(width="600px")
    )
    
    # Creates the W&B checkbox
    wandb_checkbox = widgets.Checkbox(
        description="Update W&B run",
        value=True
    )
    
    # Creates the external evaluation checkbox
    external_checkbox = widgets.Checkbox(
        description="External evaluation",
        value=False
    )
    
    # Creates the evaluate button
    evaluate_button = widgets.Button(
        description="Evaluate Model",
        button_style="success"
    )
    
    # Creates the output area
    output_area = widgets.Output(
        layout=widgets.Layout(height="auto", overflow="auto")
    )
    
    # Defines the on click function
    def on_evaluate_click(button):
        if not model_dropdown.value:
            with output_area:
                clear_output(wait=True)
                console.print("[red]Please select a model first![/red]")
                return
        
        # Handles the evaluation click event
        _handle_evaluation_click(
            model_name=model_dropdown.value,
            update_wandb=wandb_checkbox.value,
            include_external=external_checkbox.value,
            project_root=project_root,
            output_area=output_area
        )
    
    # Binds the on click function to the evaluate button
    evaluate_button.on_click(on_evaluate_click)
    
    # Returns the widget
    return widgets.VBox([
        widgets.HTML("<h3>Model Evaluation</h3>"),
        widgets.HTML("<p><strong>Validation Performance:</strong> Reported on validation set (optimistically biased due to model selection)</p>"),
        widgets.HTML("<p><strong>External Evaluation:</strong> Zero-shot transfer on GermEval-2018 (unbiased assessment)</p>"),
        widgets.HBox([model_dropdown]),
        widgets.HBox([wandb_checkbox, external_checkbox]),
        evaluate_button,
        output_area
    ])
