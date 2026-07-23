# Competition-Solution/src/model_comparison_utils.py

"""
Model Comparison Utilities

Compares evaluated models using bootstrap-based significance tests and visual summaries.
This module aligns with `single_model_evaluation_utils` by consuming its result objects and
providing:
  - Pairwise bootstrap differences with confidence intervals across metrics
  - McNemar's test for binary subtasks to compare paired predictions

Author: Samuel Ruairí Bullard
Project: Evaluating Cost-Sensitive Loss Functions for Transformer-Based German Harmful Content Detection
Date: January 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from rich.console import Console
from rich.table import Table        
import ipywidgets as widgets
from IPython.display import clear_output

# Local imports
from visualizations_compare import plot_model_comparison as viz_plot_model_comparison
from visualizations_compare import plot_performance_matrix as viz_plot_performance_matrix
from single_model_evaluation_utils import evaluate_model, discover_models
from plot_theme import apply_plot_theme

console = Console()

# Performs a bootstrap-based significance test between two models on a selected metric
def compare_models_bootstrap(results1, results2, metric='macro_f1', n_bootstraps=1000, confidence_level=0.95):
    true_labels = results1['true_labels']
    predicted_labels_model1 = results1['predictions']
    predicted_labels_model2 = results2['predictions']
    
    # Defines the metric function
    metric_functions = {
        'accuracy': accuracy_score,
        'macro_f1': lambda sample_true_labels, sample_predicted_labels: f1_score(sample_true_labels, sample_predicted_labels, average='macro'),
        'macro_precision': lambda sample_true_labels, sample_predicted_labels: precision_score(sample_true_labels, sample_predicted_labels, average='macro', zero_division=0),
        'macro_recall': lambda sample_true_labels, sample_predicted_labels: recall_score(sample_true_labels, sample_predicted_labels, average='macro', zero_division=0),
        'matthews_corrcoef': matthews_corrcoef
    }
    
    if metric not in metric_functions:
        raise ValueError(f"Unsupported metric: {metric}")
    
    metric_func = metric_functions[metric]
    
    # Defines the bootstrap statistic as the metric difference
    def difference_stat(sample_true_labels, sample_predicted_labels_model1, sample_predicted_labels_model2):
        model1_score = metric_func(sample_true_labels, sample_predicted_labels_model1)
        model2_score = metric_func(sample_true_labels, sample_predicted_labels_model2)
        return model1_score - model2_score
    
    # Performs the bootstrap test
    bootstrap_data = (true_labels, predicted_labels_model1, predicted_labels_model2)
    bootstrap_result = bootstrap(
        bootstrap_data,
        difference_stat,
        n_resamples=n_bootstraps,
        confidence_level=confidence_level,
        random_state=42,
        paired=True,
        method='BCa'
    )
    
    # Calculates the observed difference
    observed_difference = difference_stat(true_labels, predicted_labels_model1, predicted_labels_model2)
    
    # Checks whether the confidence interval contains 0
    confidence_interval_lower = bootstrap_result.confidence_interval.low
    confidence_interval_upper = bootstrap_result.confidence_interval.high
    is_significant = not (confidence_interval_lower <= 0 <= confidence_interval_upper)
    
    return {
        'metric': metric,
        'observed_difference': float(observed_difference),
        'confidence_interval': {'lower': float(confidence_interval_lower), 'upper': float(confidence_interval_upper)},
        'is_significant': is_significant,
        'confidence_level': confidence_level,
        'interpretation': 'Model 1 significantly better' if is_significant and observed_difference > 0 
                        else 'Model 2 significantly better' if is_significant and observed_difference < 0
                        else 'No significant difference'
    }


# Runs pairwise bootstrap comparisons across models and metrics
def compare_multiple_models(results_dict, metrics=['accuracy', 'macro_f1', 'macro_precision', 'macro_recall'], 
                           n_bootstraps=1000, confidence_level=0.95):
    model_names = list(results_dict.keys())
    comparisons = []
    
    # Performs pairwise comparisons
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:
                for metric in metrics:
                    try:
                        comparison = compare_models_bootstrap(
                            results_dict[model1], results_dict[model2], 
                            metric=metric, n_bootstraps=n_bootstraps, 
                            confidence_level=confidence_level
                        )
                        comparison.update({
                            'model1': model1,
                            'model2': model2
                        })
                        comparisons.append(comparison)
                    except Exception as e:
                        console.print(f"[red]Error comparing {model1} vs {model2} on {metric}: {e}[/red]")
    
    return pd.DataFrame(comparisons)


# Prints a summary table of pairwise model comparisons
def print_comparison_summary(comparison_df):
    if comparison_df.empty:
        console.print("[yellow]No comparisons available[/yellow]")
        return
    
    # Shows the Codabench baselines for context again
    console.print("\n[bold blue]Codabench Contest Baselines (for reference):[/bold blue]")
    baseline_table = Table()
    baseline_table.add_column("Subtask", style="cyan")
    baseline_table.add_column("Baseline Macro-F1", style="yellow")
    baseline_table.add_column("Approach", style="dim")
    
    baselines_info = {
        "C2A": ("0.590", "Gradient boosting + SentenceBert + undersampling"),
        "DBO": ("0.470", "Cost-sensitive SVM + TF-IDF bag-of-phrases"),
        "VIO": ("0.690", "Qwen2.5-32B few-shot")
    }
    
    # Adds the baselines to the table
    for subtask, (score, approach) in baselines_info.items():
        baseline_table.add_row(subtask, score, approach)
    
    console.print(baseline_table)
    console.print()
    
    table = Table(title="Statistical Model Comparison Summary")
    table.add_column("Model 1", style="cyan")
    table.add_column("Model 2", style="cyan")
    table.add_column("Metric", style="green")
    table.add_column("Difference", style="yellow")
    table.add_column("95% CI", style="magenta")
    table.add_column("Significant", style="red")
    
    for _, row in comparison_df.iterrows():
        significance_label = "YES" if row['is_significant'] else "NO"
        significance_color = "green" if row['is_significant'] else "red"
        
        table.add_row(
            row['model1'],
            row['model2'],
            row['metric'],
            f"{row['observed_difference']:+.4f}",
            f"[{row['confidence_interval']['lower']:.4f}, {row['confidence_interval']['upper']:.4f}]",
            f"[{significance_color}]{significance_label}[/{significance_color}]"
        )
    
    console.print(table)


# Plots bar charts comparing models across selected metrics
def plot_model_comparison(results_dict, metrics=['accuracy', 'macro_f1'], figsize=(12, 6)):
    return viz_plot_model_comparison(results_dict, metrics=metrics, figsize=figsize)


# Plots a heatmap performance matrix across models and metrics
def plot_performance_matrix(results_dict, metrics=['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']):
    return viz_plot_performance_matrix(results_dict, metrics=metrics)


# Runs McNemar's test for binary classification predictions of two models
def mcnemar_test(results1, results2, exact=False, correction=True):
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

    true_labels = results1['true_labels']
    predicted_labels_model1 = results1['predictions']
    predicted_labels_model2 = results2['predictions']

    # Checks whether the task is binary
    if len(np.unique(true_labels)) != 2:
        raise ValueError("McNemar's test is only applicable to binary classification")

    # Creates the contingency table
    model1_is_correct = (predicted_labels_model1 == true_labels)
    model2_is_correct = (predicted_labels_model2 == true_labels)

    both_correct = int(np.sum(model1_is_correct & model2_is_correct))
    model1_only = int(np.sum(model1_is_correct & ~model2_is_correct))
    model2_only = int(np.sum(~model1_is_correct & model2_is_correct))
    both_wrong  = int(np.sum(~model1_is_correct & ~model2_is_correct))

    table = [[both_correct, model1_only],
             [model2_only, both_wrong]]

    mcnemar_result = sm_mcnemar(table, exact=exact, correction=correction)
    return {
        'statistic': float(mcnemar_result.statistic),
        'p_value': float(mcnemar_result.pvalue),
        'is_significant': mcnemar_result.pvalue < 0.05,
        'contingency_table': table
    }


# Discovers the subtasks and models available
def _build_model_options(discovered):
    # Builds (display, value) option tuples grouped by subtask
    options = []
    for subtask, models in (discovered or {}).items():
        for model in models:
            options.append((f"{subtask.upper()} | {model['name']}", model['name']))
    return sorted(options, key=lambda x: x[0])


# Creates the model comparison widget
def create_model_comparison_widget(project_root):
    apply_plot_theme()
    # Discovers the available models
    available = discover_models(project_root)

    # Creates the model selector
    model_selector = widgets.SelectMultiple(
        options=_build_model_options(available),
        description="Models:",
        layout=widgets.Layout(width="600px", height="220px")
    )

    # Creates the show plots checkbox
    show_plots = widgets.Checkbox(value=True, description="Show plots")
    run_mcnemar = widgets.Checkbox(value=True, description='McNemar (first 2)')
    save_csv = widgets.Checkbox(value=False, description='Save CSV')

    # Creates the metrics selector
    metrics_selector = widgets.SelectMultiple(
        options=['accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'matthews_corrcoef'],
        value=('accuracy', 'macro_f1', 'macro_precision', 'macro_recall'),
        description="Metrics:",
        layout=widgets.Layout(width="320px", height="150px")
    )

    # Creates the number of bootstraps slider
    bootstrap_count_slider = widgets.IntSlider(value=2000, min=200, max=5000, step=200, description='#Bootstraps')

    # Creates the confidence level slider
    confidence_level_slider = widgets.FloatSlider(value=0.95, min=0.80, max=0.99, step=0.01, description='Confidence')

    # Creates the exact McNemar checkbox
    exact_mcnemar_checkbox = widgets.Checkbox(value=False, description='Exact McNemar')
    correction_checkbox = widgets.Checkbox(value=True, description='Correction')

    # Creates the advanced options accordion
    advanced_options_accordion = widgets.Accordion(children=[widgets.VBox([
        metrics_selector,
        widgets.HBox([bootstrap_count_slider, confidence_level_slider]),
        widgets.HBox([exact_mcnemar_checkbox, correction_checkbox])
    ])])
    advanced_options_accordion.set_title(0, 'Advanced options')

    # Creates the compare button
    compare_button = widgets.Button(description="Compare Models", button_style='success')
    output_area = widgets.Output(layout=widgets.Layout(height="auto", overflow="auto"))

    # Defines the on compare function
    def on_compare(_):
        with output_area:
            clear_output(wait=True)
            selected_models = list(model_selector.value)
            if len(selected_models) < 2:
                console.print("[red]Please select at least two models.[/red]")
                return

            # Evaluates the selected models
            evaluation_results_by_model = {}
            for model_name in selected_models:
                try:
                    evaluation_result = evaluate_model(
                        model_name=model_name,
                        project_root=project_root,
                        update_wandb=False,
                        include_confidence_intervals=False,
                        include_error_analysis=False,
                        include_external_eval=False
                    )
                    evaluation_results_by_model[model_name] = evaluation_result
                except Exception as e:
                    console.print(f"[red]Failed to evaluate {model_name}: {e}")

            if len(evaluation_results_by_model) < 2:
                console.print("[red]Need two successful evaluations to compare.[/red]")
                return

            # Performs the pairwise bootstrap comparisons and prints the summary table
            comparison_dataframe = compare_multiple_models(
                evaluation_results_by_model,
                metrics=list(metrics_selector.value) or ['accuracy', 'macro_f1'],
                n_bootstraps=int(bootstrap_count_slider.value),
                confidence_level=float(confidence_level_slider.value)
            )
            print_comparison_summary(comparison_dataframe)

            if save_csv.value:
                report_dir = (project_root / "reports")
                report_dir.mkdir(exist_ok=True)
                output_path = report_dir / "statistical_model_comparison.csv"
                comparison_dataframe.to_csv(output_path, index=False)
                console.print(f"[green]Saved CSV:[/green] {output_path}")

            # Plots the model comparison and performance matrix
            if show_plots.value:
                plot_model_comparison(evaluation_results_by_model, metrics=['accuracy', 'macro_f1'])
                plot_performance_matrix(evaluation_results_by_model, metrics=list(metrics_selector.value))

            # Runs McNemar's test on the first two binary models
            if run_mcnemar.value:
                first_two_model_names = list(evaluation_results_by_model.keys())[:2]
                first_model_results = evaluation_results_by_model[first_two_model_names[0]]
                second_model_results = evaluation_results_by_model[first_two_model_names[1]]
                if len(np.unique(first_model_results['true_labels'])) == 2 == len(np.unique(second_model_results['true_labels'])):
                    mcnemar_results = mcnemar_test(
                        first_model_results,
                        second_model_results,
                        exact=exact_mcnemar_checkbox.value,
                        correction=correction_checkbox.value
                    )
                    console.print(
                        f"\nMcNemar: {first_two_model_names[0]} vs {first_two_model_names[1]} -> "
                        f"Statistic: {mcnemar_results['statistic']:.4f}, "
                        f"p-value: {mcnemar_results['p_value']:.6f}  "
                        f"(Significant: {'Yes' if mcnemar_results['is_significant'] else 'No'})"
                    )
                else:
                    console.print("[yellow]Skipping McNemar (requires binary labels for both models).")

    compare_button.on_click(on_compare)

    # Returns the widget
    return widgets.VBox([
        widgets.HTML("<h3>Model Comparison</h3>"),
        widgets.HTML("<p>Select two or more trained runs to compare. Macro‑F1 is primary.</p>"),
        widgets.HBox([model_selector]),
        widgets.HBox([show_plots, save_csv, run_mcnemar]),
        compare_button,
        advanced_options_accordion,
        output_area
    ])
