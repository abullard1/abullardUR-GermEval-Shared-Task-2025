**Evaluating Cost-Sensitive Loss Functions for Transformer-Based German Harmful Content Detection**
**GermEval 2025: A Shared Task for the Detection of Harmful Content on Social Media**
_**Author: Samuel Ruairí Bullard**_
_**Institution: University of Regensburg**_
_**Date: March 2026**_

This repository contains the code and data for the practical component of the project. The work investigates class-weighted and focal loss strategies for imbalanced harmful content detection, using ModernGBERT as the base model and the GermEval 2025 shared task as the evaluation benchmark.

### The GermEval 2025 Shared Task

The [GermEval 2025 Shared Task](https://www.codabench.org/competitions/4963/#/) is a competition organised by the University of Stuttgart and the University of Mannheim. The task is to detect harmful content on social media. The competition is organised into three subtasks:

  1. **Subtask 1 (C2A):** Binary detection of **calls to action**, i.e. calls for risky actions (e.g. criminal offences, demonstrations with escalation potential)
  2. **Subtask 2 (DBO):** Fine-grained classification into four categories of **attacks against the free and democratic basic order** of the Federal Republic of Germany
  3. **Subtask 3 (VIO):** Binary detection of disturbing, **violence-related** statements in tweets

### Approach

For each subtask, three training strategies were compared:

1. **Baseline (CE):** Standard cross-entropy loss with no class balancing
2. **Class-Weighted Cross-Entropy (CWCE):** Cross-entropy with inverse-frequency class weights
3. **Class-Weighted Focal Loss (CW+FL):** Focal loss with class weights and a tunable gamma parameter

All nine models (3 subtasks × 3 strategies) use [ModernGBERT](https://huggingface.co/LSX-UniWue/ModernGBERT_134M) as the base encoder. Hyperparameters were optimised via Bayesian sweeps tracked in Weights & Biases. The class-weighted cross-entropy approach achieved the best macro-F1 across all three subtasks and was selected for the competition submission.

### Repository Structure

```
Competition-Solution/
├── baseline/                # Organiser-provided baseline systems (gradient boosting, SVM)
├── configs/                 # YAML configuration files for all experiments
│   ├── base.yaml            # Global training arguments, W&B settings, paths
│   ├── subtask1_call2action/
│   ├── subtask2_attacks_on_dbo/
│   ├── subtask3_violence/
│   └── mtl/                 # Multi-task learning config (exploratory, not used in submission)
├── data/                    # Dataset files
│   ├── raw/                 # Original CSV files from the shared task
│   ├── processed/           # HuggingFace datasets (tokenised, split)
│   └── evaluation/          # External benchmark (GermEval 2018) for zero-shot transfer evaluation
├── models/
│   ├── base_models/         # Downloaded pretrained checkpoints
│   └── finetuned_models/    # Trained model checkpoints (per subtask and strategy)
├── notebooks/
│   ├── train/               # Training phase notebooks
│   │   ├── 00_data_profiling/
│   │   ├── 01_data_preprocessing/
│   │   ├── 02_model_training/
│   │   └── 03_model_evaluation/
│   └── trial/               # Trial phase notebooks (data profiling, preprocessing, baseline training)
├── prediction_submissions/  # Generated competition submission CSVs and zips
├── analysis_exports/        # Exported evaluation evidence and thesis analysis data
├── assets/images/           # Static image assets
├── src/                     # Shared utility modules used by the notebooks
│   ├── config_utils.py              # YAML config loading and merging
│   ├── data_utils.py                # Data loading, preprocessing, analysis
│   ├── single_model_evaluation_utils.py  # Per-model evaluation and metrics
│   ├── model_comparison_utils.py    # Cross-model comparison utilities
│   ├── evaluation_augment.py        # Post-hoc analyses (calibration, error overlap, confidence, vocabulary)
│   ├── wandb_utils.py               # W&B integration (runs, sweeps, artifacts)
│   ├── visualizations_single.py     # Single-model plots
│   ├── visualizations_compare.py    # Comparison plots
│   ├── plot_theme.py                # Matplotlib theme
│   └── generate_submission_results.py  # Competition submission file generation
├── pyproject.toml           # Python dependencies (uv / PEP 621)
├── uv.lock                  # Locked dependency versions (cross-platform)
├── setup.sh                 # Automated environment setup script (uses uv)
└── README.md                # This file
```

### Data

The datasets are derived from posts and comments on the public German-language Twitter network of a right-wing extremist group. These tweets were collected by members of Mittweida University of Applied Sciences from July 2016 to December 2020. In October 2024, four university members with a forensic background annotated each tweet. The final datasets for each subtask contain all tweets (between 9,000 and 11,000) for which a majority decision of the four annotators could be made.

The data is stored in the `data` directory:
- `raw/`: Original CSV files as distributed by the shared task organisers
- `processed/`: HuggingFace `datasets` objects (tokenised and split into train/validation/test)
- `evaluation/`: External GermEval 2018 test set used for zero-shot transfer evaluation

### Get Started

**Prerequisites:** Linux (tested on Ubuntu 22.04 and Pop!_OS 22.04), Python 3.10, an NVIDIA GPU with CUDA support.

1. Clone the repository and navigate to the Competition-Solution directory.
2. Run the setup script to create a virtual environment and install all dependencies:
   ```bash
   bash setup.sh
   ```
3. Create a `.env` file with your Weights & Biases API key (required for experiment tracking):
   ```
   WANDB_API_KEY="your-api-key-here"
   ```
4. The notebooks in `notebooks/train/` are meant to be run in order:
   - `00_data_profiling/` — Exploratory data analysis
   - `01_data_preprocessing/` — Tokenisation and dataset preparation
   - `02_model_training/` — Model fine-tuning (baseline, class-weighted, focal loss)
   - `03_model_evaluation/` — Evaluation, comparison, and submission file generation

### System and Versions

| Component | Specification                           |
|-----------|---------------------------------------- |
| OS        | Pop!_OS 22.04 LTS                       |
| Python    | 3.10.12                                 |
| Shell     | bash 5.1.16                             |
| CPU       | AMD Ryzen 9 5900X 12-Core @ 3.700GHz   |
| GPU       | NVIDIA GeForce RTX 3080 Ti (12GB VRAM)  |
| Memory    | ~32GB                                   |
