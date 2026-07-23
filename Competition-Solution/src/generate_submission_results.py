# Competition-Solution/src/generate_submission_results.py

"""
Submission Generation Utilities

This module provides utilities for generating the competition submissions
for the GermEval 2025 Shared Task. It handles model inference on the hidden test-set data,
result formatting, and submission file creation in the required format.

Author: Samuel Ruairí Bullard
Project: Evaluating Cost-Sensitive Loss Functions for Transformer-Based German Harmful Content Detection
Date: January 2026
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from pathlib import Path
from rich.console import Console
import ipywidgets as widgets
from IPython.display import clear_output
from datasets import load_from_disk
from datetime import datetime
import zipfile
import shutil

from single_model_evaluation_utils import discover_models, get_eval_config, CLASS_NAMES

console = Console()


# Generates a submission CSV for a given fine-tuned model
def generate_submission(model_name, project_root, output_dir="prediction_submissions", team_name="abullardUR", run_number=1):
    config = get_eval_config(model_name, project_root)

    # Defines the device to use for the model (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loads the processor and model
    processor = AutoProcessor.from_pretrained(config["model_path"], trust_remote_code=True)
    processor.add_special_tokens({"additional_special_tokens": config["special_tokens"]})

    # Loads the model with half precision if CUDA is available, otherwise uses full precision (CPU)
    if torch.cuda.is_available():
        model = AutoModelForSequenceClassification.from_pretrained(config["model_path"], torch_dtype=torch.float16)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(config["model_path"])

    # Resizes the token embeddings to the length of the processor
    model.resize_token_embeddings(len(processor), mean_resizing=True)

    # Moves the model to the device and sets it to evaluation mode
    model.to(device).eval()

    # Loads the pre-tokenized dataset from the same path used for training/evaluation (i.e. the test set)
    tokenized = load_from_disk(str(config["dataset_path"]))
    test_dataset = tokenized["test"]

    # Ensures the test split has no label column (hidden test set needs to be unlabeled like raw CSV)
    # We drop the "labels" column to prevent the collator from tensorizing None values during prediction
    if "labels" in test_dataset.column_names:
        test_dataset = test_dataset.remove_columns(["labels"])

    # Gets the ids from the test dataset
    test_ids = test_dataset["id"]

    # Defines the training arguments for the model
    args = TrainingArguments(
        output_dir="./tmp_submission",
        per_device_eval_batch_size=16,
        dataloader_drop_last=False,
        report_to="none",
        disable_tqdm=True,
    )

    # Defines the trainer for the model with the standard data collator
    trainer = Trainer(model=model, args=args, data_collator=DataCollatorWithPadding(processor))

    # Runs the predictions on the test set
    predictions_result = trainer.predict(test_dataset, ignore_keys=["id", "text", "description", "labels"])
    predictions = np.argmax(predictions_result.predictions, axis=1)

    # Formats the predictions based on the subtask
    subtask = config["subtask_id"]

    if subtask in ["c2a", "vio"]:
        # Formats predictions as TRUE/FALSE for binary subtasks (e.g. C2A and Violence)
        formatted = ["FALSE" if p == 0 else "TRUE" for p in predictions]  # 0 = False, 1 = True
        pred_col = subtask
    elif subtask == "dbo":
        # Maps prediction indices to DBO class names (lowercased) (e.g. DBO)
        dbo_names = []
        for c in CLASS_NAMES["dbo"]:
            dbo_names.append(c.lower())
        formatted = []
        for p in predictions:
            formatted.append(dbo_names[p])
        pred_col = "dbo"
    else:
        raise ValueError(f"Unknown subtask: {subtask}")
    
    # Creates the submission dataframe with the correct column names as defined by the organizers
    submission_df = pd.DataFrame({"id": test_ids, pred_col: formatted})
    
    # Ensures the structured output directory exists: prediction_submissions/run_<run_number>/<subtask>/<model_name>
    model_dir = Path(output_dir) / f"run_{run_number}" / subtask / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Creates the filename for the submission CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{team_name}{run_number}_{subtask}_{timestamp}.csv"

    # Saves the submission dataframe to a CSV file
    output_path = model_dir / filename
    submission_df.to_csv(output_path, index=False, sep=";", quoting=1)  # Semicolon delimiter, quote non-numeric values

    # Deletes the model and processor to free up memory again, ready for the next model and evaluation/submission run
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    console.print(f"[green] Saved: {output_path}")

    return str(output_path)

# Generates submissions for all discovered models and returns their output file paths
def generate_all_submissions(project_root, output_dir="prediction_submissions", team_name="abullardUR", run_number=1):
    models = discover_models(project_root)

    # Initializes the paths dictionary
    paths = {}

    # Loops through the subtasks and models
    for subtask, model_list in models.items():
        # Loops through the models for the subtask
        for model in model_list:

            # Prints the subtask and model name
            console.print(f"\n[blue]Processing: {subtask.upper()} | {model['name']}[/blue]")

            # Generates the submission for the model
            path = generate_submission(model["name"], project_root, output_dir, team_name, run_number)

            # Adds the path to the paths dictionary
            paths[model["name"]] = path

    console.print(f"\n[bold green] Generated {len(paths)} submission files[/bold green]")
    return paths

# Validates a submission CSV against expected column names and value sets
# Just a sanity check, since our first submission failed and I really didn't want to waste another one
# of my 3 maximum submission attempts that I am permitted to make
def validate_submission(submission_path, expected_subtask=None):
    try:
        # Reads the csv with the semicolon delimiter and keeps the values as strings
        submission_dataframe = pd.read_csv(submission_path, sep=";", dtype=str)
        
        # Initializes the issues list
        issues = []
        
        # Checks for the required columns based on subtask
        if "id" not in submission_dataframe.columns:
            issues.append("Missing 'id' column")
        
        # Defines the expected prediction column based on subtask
        expected_pred_column = expected_subtask if expected_subtask else "prediction"

        # Checks for the expected prediction column
        if expected_pred_column not in submission_dataframe.columns:
            issues.append(f"Missing '{expected_pred_column}' column")
            return {
                "status": "invalid",
                "issues": issues,
                "num_samples": len(submission_dataframe),
                "unique_predictions": []
            }
        
        # Checks for missing values in the id and prediction columns
        if submission_dataframe["id"].isnull().any():
            issues.append("Missing values in 'id' column")
        if submission_dataframe[expected_pred_column].isnull().any():
            issues.append(f"Missing values in '{expected_pred_column}' column")
        
        # Checks for invalid prediction values based on subtask
        if expected_subtask:
            unique_predictions = set(submission_dataframe[expected_pred_column].unique())
            
            if expected_subtask in ["c2a", "vio"]:
                # Binary classification: TRUE or FALSE (e.g. C2A and Violence)
                valid_values = {"TRUE", "FALSE"}
                if not unique_predictions.issubset(valid_values):
                    issues.append(f"Invalid prediction values: {unique_predictions - valid_values}")
            
            elif expected_subtask == "dbo":
                # Multi-class classification: nothing, criticism, agitation, subversive (e.g. DBO)
                valid_values = {"nothing", "criticism", "agitation", "subversive"}
                if not unique_predictions.issubset(valid_values):
                    issues.append(f"Invalid prediction values: {unique_predictions - valid_values}")
        
        # Defines the status based on the issues
        status = "valid" if not issues else "invalid"

        # Returns the validation results
        return {
            "status": status,
            "issues": issues,
            "num_samples": len(submission_dataframe),
            "unique_predictions": list(submission_dataframe[expected_pred_column].unique())
        }
        
    except Exception as e:
        # Returns the validation results
        return {
            "status": "error",
            "issues": [f"Error reading file: {e}"],
            "num_samples": 0,
            "unique_predictions": []
        }

# Creates a competition-ready zip file containing the correctly named submission CSV files
def create_competition_zip(submission_files, team_name="abullardUR", run_number=1, output_dir="prediction_submissions"):
    # Places the zip inside prediction_submissions/run_<run_number>/
    output_path = Path(output_dir) / f"run_{run_number}"

    # Ensures the output directory exists, creates the run_<run_number> folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Defines the zip filename based on the team name and run number
    zip_filename = f"{team_name}{run_number}.zip"
    zip_path = output_path / zip_filename
    
    # Creates a temporary directory for renamed files
    temp_dir = output_path / f"temp_{team_name}{run_number}"
    temp_dir.mkdir(exist_ok=True)
    
    # Attempts to create the zip file
    try:
        renamed_files = []
        
        # Copies and renames files according to the competition format: team_name<run_number>_<subtask_id>.csv
        for subtask_id, file_path in submission_files.items():
            # Checks if the file exists
            if Path(file_path).exists():
                new_filename = f"{team_name}{run_number}_{subtask_id}.csv"
                new_path = temp_dir / new_filename

            # Copies the file to the temporary directory and adds it to the list of renamed files
            shutil.copy2(file_path, new_path)
            renamed_files.append(new_path)

            console.print(f"[green][OK] Prepared: {new_filename}[/green]")
        
        # Creates the zip file with the competition format
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:

            # Loops through the renamed files and adds them to the zip file
            for file_path in renamed_files:
                zipf.write(file_path, file_path.name)
        
        console.print(f"[bold green][OK] Competition zip created: {zip_path}[/bold green]")
        console.print(f"[cyan]Contents:[/cyan]")

        # Prints the contents of the zip file
        for subtask_id in submission_files.keys():
            console.print(f"  - {team_name}{run_number}_{subtask_id}.csv")
        
        return str(zip_path)
    
    # Cleans up the temporary directory after the zip file is created
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

# Creates an interactive Jupyter widget to generate, validate, and zip submissions
def create_submission_widget(project_root):
    # Discovers the models
    models = discover_models(project_root)

    # Builds the per-subtask dropdowns (One model per subtask)
    def build_options_for_subtask(subtask_id):
        options = [("None", "")]
        for model in models.get(subtask_id, []):
            options.append((model["name"], model["name"]))
        return options

    # Creates the C2A dropdown
    c2a_dropdown = widgets.Dropdown(
        options=build_options_for_subtask("c2a"),
        description="C2A:",
        layout=widgets.Layout(width="600px")
    )

    # Creates the DBO dropdown
    dbo_dropdown = widgets.Dropdown(
        options=build_options_for_subtask("dbo"),
        description="DBO:",
        layout=widgets.Layout(width="600px")
    )

    # Creates the Violence dropdown
    vio_dropdown = widgets.Dropdown(
        options=build_options_for_subtask("vio"),
        description="VIO:",
        layout=widgets.Layout(width="600px")
    )

    # Creates the output directory text input
    output_dir_text = widgets.Text(
        value=str((Path(project_root) / "prediction_submissions")), # Path from the project root
        description="Output Dir (from project root):",
        layout=widgets.Layout(width="420px")
    )

    # Creates the team name text input
    team_name_text = widgets.Text(
        value="abullardUR",
        description="Team:",
        layout=widgets.Layout(width="180px")
    )

    # Creates the run number text input
    run_number_text = widgets.IntText(
        value=1,
        description="Run Number:",
        min=1,
        max=3,
        layout=widgets.Layout(width="150px")
    )

    # Creates the validate checkbox
    validate_checkbox = widgets.Checkbox(
        description="Validate outputs",
        value=True
    )

    # Creates the create zip checkbox
    create_zip_checkbox = widgets.Checkbox(
        description="Create competition zip",
        value=True
    )

    # Creates the generate button
    generate_button = widgets.Button(
        description="Generate Competition Submission", 
        button_style="primary"
    )
    output_area = widgets.Output()

    # Defines the on click function
    def on_click(b):
        with output_area:
            clear_output(wait=True)
            try:
                output_dir = output_dir_text.value
                team_name = team_name_text.value
                run_number = run_number_text.value
                validate = validate_checkbox.value
                create_zip = create_zip_checkbox.value

                # Collects selected models by subtask
                selections = {
                    "c2a": c2a_dropdown.value,
                    "dbo": dbo_dropdown.value,
                    "vio": vio_dropdown.value,
                }

                # Initializes the submission paths and subtask files dictionaries
                submission_paths = {}
                subtask_files = {}

                # Generates the selected submissions
                for subtask_id, model_name in selections.items():
                    if not model_name:
                        continue
                    path = generate_submission(model_name, project_root, output_dir, team_name, run_number)
                    submission_paths[model_name] = path
                    subtask_files[subtask_id] = path

                if not submission_paths:
                    console.print("[yellow]No models selected. Please choose at least one.[/yellow]")
                    return

                # Validates the submissions
                if validate:
                    console.print("\n[bold cyan]Validating submissions...[/bold cyan]")

                    # Loops through the subtask files and validates them
                    for subtask_id, file_path in subtask_files.items():
                        validation = validate_submission(file_path, subtask_id)

                        # Prints the validation status in green if valid, red if invalid
                        status_color = "green" if validation['status'] == 'valid' else "red"
                        console.print(f"[{status_color}]{subtask_id.upper()}: {validation['status']}[/{status_color}]")

                        # Prints the issues if the submission is invalid
                        if validation['issues']:
                            # Loops through the issues and prints them
                            for issue in validation['issues']:
                                console.print(f"  - {issue}")

                # Creates the competition zip if requested and there are any files to zip
                if create_zip and subtask_files:
                    console.print("\n[bold cyan]Creating competition zip...[/bold cyan]")
                    zip_path = create_competition_zip(
                        subtask_files, 
                        team_name, 
                        run_number, 
                        output_dir
                    )
                    console.print(f"[bold yellow] Ready for upload to Codabench: {zip_path}[/bold yellow]")

                console.print(f"\n[bold green] Competition submission ready![/bold green]")

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                import traceback
                traceback.print_exc()

    # Binds the click handler to the generate button
    generate_button.on_click(on_click)
    
    # Returns the widget
    return widgets.VBox([
        widgets.HTML("<h3>Competition Submission Generator</h3>"),
        widgets.HBox([c2a_dropdown]),
        widgets.HBox([dbo_dropdown]),
        widgets.HBox([vio_dropdown]),
        widgets.HBox([output_dir_text, team_name_text, run_number_text]),
        widgets.HBox([validate_checkbox, create_zip_checkbox]),
        generate_button,
        output_area
    ])
