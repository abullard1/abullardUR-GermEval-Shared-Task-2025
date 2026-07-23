# Competition-Solution/src/plot_theme.py

"""
Plot Theme and Colour Constants

Shared matplotlib/seaborn style settings and colour definitions
used by the evaluation notebooks and visualisation modules.

Author: Samuel Ruairí Bullard
Project: Evaluating Cost-Sensitive Loss Functions for Transformer-Based German Harmful Content Detection
Date: January 2026
"""

import matplotlib as mpl
import seaborn as sns

# Strategy colours
STRATEGY_COLOURS = {
    "Baseline": "#56B4E9",  # sky blue
    "CWCE": "#009E73",      # bluish green
    "CW+FL": "#E69F00",     # orange
}
STRATEGY_LINESTYLES = {"Baseline": "-", "CWCE": "--", "CW+FL": ":"}
STRATEGY_MARKERS = {"Baseline": "o", "CWCE": "s", "CW+FL": "^"}

# Subtask colours
SUBTASK_COLOURS = {
    "c2a": "#5b9bd5",  # c2aAccent
    "dbo": "#d4a017",  # dboAccent
    "vio": "#c0392b",  # vioAccent
}

# Class labels for figures
CLASS_NAMES_DISPLAY = {
    "c2a": ["No Call to Action", "Call to Action"],
    "dbo": ["Nothing", "Criticism", "Agitation", "Subversive"],
    "vio": ["Non-violent", "Violent"],
}

USETEX = False

# Applies the plot style to matplotlib/seaborn
def apply_plot_theme():
    global USETEX
    sns.set_theme(style="whitegrid", font_scale=1.2)

    base_params = {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "serif",
        "lines.linewidth": 1.5,
    }

    # Tries LaTeX rendering for serif fonts (Latin Modern)
    try:
        import shutil
        if shutil.which("latex") is None:
            raise RuntimeError("latex binary not found")
        import matplotlib.pyplot as plt
        mpl.rcParams.update({
            "text.usetex": True,
            "font.serif": ["Latin Modern Roman"],
            **base_params,
        })

        # Quick check that the LaTeX actually works
        figure, axis = plt.subplots(figsize=(1, 1))
        axis.set_title(r"$\mathrm{test}$")
        figure.canvas.draw()
        plt.close(figure)
        USETEX = True

    except Exception:
        # Falls back to system serif fonts if LaTeX is not available
        mpl.rcParams.update({**base_params, "text.usetex": False})
        USETEX = False
