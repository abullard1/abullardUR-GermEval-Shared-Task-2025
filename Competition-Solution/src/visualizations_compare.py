# Competition-Solution/src/visualizations_compare.py

"""
Model Comparison Visualization Utilities

Provides plotting helpers for the model comparisons.

Author: Samuel Ruairí Bullard
Project: Evaluating Cost-Sensitive Loss Functions for Transformer-Based German Harmful Content Detection
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Plots a bar chart comparing models across selected metrics
def plot_model_comparison(evaluation_results_by_model, metrics=["accuracy", "macro_f1"], figsize=(12, 6)):
    n_metrics = len(metrics)
    
    figure, axis_array = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axis_array = [axis_array]
    
    # Iterates through the metrics
    for i, metric in enumerate(metrics):
        values = []
        model_names = []
        
        # Iterates through the model names
        for model_name, evaluation_results in evaluation_results_by_model.items():
            if metric in evaluation_results["metrics"]:
                values.append(evaluation_results["metrics"][metric])
                model_names.append(model_name)
        
        if values:
            bars = axis_array[i].bar(range(len(values)), values, alpha=0.7)
            axis_array[i].set_title(f"{metric.replace('_', ' ').title()}")
            axis_array[i].set_xticks(range(len(model_names)))
            axis_array[i].set_xticklabels(model_names, rotation=45, ha="right")
            axis_array[i].set_ylabel("Score")
            axis_array[i].grid(True, alpha=0.3)
            
            for bar, value in zip(bars, values):
                axis_array[i].text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom"
                )
    
    plt.tight_layout()
    plt.show()


# Plots a heatmap performance matrix across models and metrics
def plot_performance_matrix(evaluation_results_by_model, metrics=["accuracy", "macro_f1", "macro_precision", "macro_recall"]):
    # Gets the model names
    model_names = list(evaluation_results_by_model.keys())
    
    # Creates the performance matrix
    performance_matrix = []

    # Iterates through the model names
    for model_name in model_names:
        row = []

        # Iterates through the metrics
        for metric in metrics:
            # Checks if the metric is in the results
            if metric in evaluation_results_by_model[model_name]["metrics"]:
                row.append(evaluation_results_by_model[model_name]["metrics"][metric])
            else:
                row.append(np.nan)
        performance_matrix.append(row)
    
    # Formats the columns
    formatted_columns = []

    for metric in metrics:
        formatted_columns.append(metric.replace("_", " ").title())
    performance_dataframe = pd.DataFrame(
        performance_matrix,
        index=model_names,
        columns=formatted_columns
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(performance_dataframe, annot=True, fmt=".3f", cmap="Blues", 
                cbar_kws={"label": "Score"})
    plt.title("Model Performance Matrix")
    plt.tight_layout()
    plt.show()
