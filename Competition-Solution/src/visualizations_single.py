# Competition-Solution/src/visualizations_single.py

"""
Data Visualization Utilities

Provides helper methods for generating visualizations for single model evaluations.

Author: Samuel Ruairí Bullard
Project: Evaluating Cost-Sensitive Loss Functions for Transformer-Based German Harmful Content Detection
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve as sk_roc_curve
from sklearn.metrics import precision_recall_curve as sk_pr_curve


# Plots a confusion matrix
def plot_confusion_matrix(true_labels, predictions, class_names, axis):
    confusion_matrix_values = sk_confusion_matrix(true_labels, predictions)
    sns.heatmap(
        confusion_matrix_values,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis
    )
    axis.set_title("Confusion Matrix")
    axis.set_ylabel("True")
    axis.set_xlabel("Predicted")


# Plots a ROC curve
def plot_roc_curve(true_labels, probabilities, roc_auc, axis):
    false_positive_rate, true_positive_rate, _ = sk_roc_curve(true_labels, probabilities[:, 1])
    axis.plot(false_positive_rate, true_positive_rate, label=f"ROC (AUC = {roc_auc:.3f})", lw=2)
    axis.plot([0, 1], [0, 1], "k--", alpha=0.5)
    axis.set_xlim([0, 1])
    axis.set_ylim([0, 1.05])
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve")
    axis.legend()
    axis.grid(True, alpha=0.3)


# Plots a precision-recall curve
def plot_precision_recall_curve(true_labels, probabilities, avg_precision, axis):
    precision_values, recall_values, _ = sk_pr_curve(true_labels, probabilities[:, 1])
    axis.plot(recall_values, precision_values, label=f"PR (AP = {avg_precision:.3f})", lw=2)
    axis.set_xlim([0, 1])
    axis.set_ylim([0, 1.05])
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-Recall Curve")
    axis.legend()
    axis.grid(True, alpha=0.3)


# Creates the visualizations for a single model evaluation
def create_visualizations(results, class_names):
    metrics = results["metrics"]
    predictions = results["predictions"]
    probabilities = results["probabilities"]
    true_labels = results["true_labels"]

    # Checks if the subtask is binary
    is_binary = ("roc_auc" in metrics)
    
    # Sets the number of plots to 3 if the subtask is binary, otherwise 1
    n_plots = 3 if is_binary else 1

    # Creates the figure and axes
    figure, axis_array = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    axis_array = np.atleast_1d(axis_array)

    # Plots the confusion matrix
    plot_confusion_matrix(true_labels, predictions, class_names, axis_array[0])

    # Plots the ROC curve and precision-recall curve if the subtask is binary
    if is_binary:
        plot_roc_curve(true_labels, probabilities, metrics['roc_auc'], axis_array[1])
        plot_precision_recall_curve(true_labels, probabilities, metrics['avg_precision'], axis_array[2])

    plt.tight_layout()
    plt.show()
