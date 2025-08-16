# GermEval 2025: Fine-tuning ModernGBERT for Harmful Content Detection

[![arXiv](https://img.shields.io/badge/arXiv-2505.13136-b31b1b.svg)](https://arxiv.org/abs/[SUBMISSION_PAPER_LINK])
[![License: RAIL-M](https://img.shields.io/badge/License-RAIL--M-blue.svg)](https://huggingface.co/LSX-UniWue/ModernGBERT_134M/blob/main/license.md)
[![Competition](https://img.shields.io/badge/GermEval-2025-green.svg)](https://www.codabench.org/competitions/4963/)

> **abullardUR@GermEval Shared Task 2025**: Fine-tuning ModernGBERT on Highly Imbalanced German Social Media for Harmful Content Detection

This repository contains the code and information for the [GermEval 2025 Shared Task](https://www.codabench.org/competitions/4963/) submission on Harmful Content Detection in German Social Media. The submission focuses on two approaches for handling class imbalance in highly imbalanced datasets through different loss calculation methods.

## 🎯 Task Overview

The submission covers all three subtasks in the GermEval 2025 Shared Task:

| Subtask | Task | Type | Description |
|---------|------|------|-------------|
| **C2A** | Call to Action | Binary | Detection of calls for risky actions |
| **DBO** | Attacks on Democratic Basic Order | Multi-class (4) | Classification of statements against democratic principles |
| **VIO** | Violence Detection | Binary | Detection of violence-related content |

## 📊 Results

### Performance Summary (Macro-F1)

| Method | C2A | DBO | VIO |
|--------|-----|-----|-----|
| Baseline (organizers) | 0.59 | 0.47 | 0.69 |
| **ModernGBERT + Class Weights (ours)** | **0.82** | **0.63** | **0.82** |
| **ModernGBERT + Weights + Focal (ours)** | **0.82** | **0.56** | **0.81** |

### Competition Rankings

- **C2A**: 4/9 teams
- **DBO**: 6/9 teams  
- **VIO**: 2/8 teams

### Performance Visualization

#### Model Comparison Chart
![Performance Comparison](https://quickchart.io/chart?c=%7B%22type%22%3A%22bar%22%2C%22data%22%3A%7B%22labels%22%3A%5B%22C2A%22%2C%22DBO%22%2C%22VIO%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22Baseline%22%2C%22data%22%3A%5B0.59%2C0.47%2C0.69%5D%2C%22backgroundColor%22%3A%22%23FF6384%22%7D%2C%7B%22label%22%3A%22CW%20%28Ours%29%22%2C%22data%22%3A%5B0.82%2C0.63%2C0.82%5D%2C%22backgroundColor%22%3A%22%2336A2EB%22%7D%2C%7B%22label%22%3A%22CW%2BFL%20%28Ours%29%22%2C%22data%22%3A%5B0.82%2C0.56%2C0.81%5D%2C%22backgroundColor%22%3A%22%234BC0C0%22%7D%5D%7D%2C%22options%22%3A%7B%22responsive%22%3Atrue%2C%22plugins%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Macro-F1%20Score%20Comparison%22%7D%7D%2C%22scales%22%3A%7B%22y%22%3A%7B%22beginAtZero%22%3Atrue%2C%22max%22%3A1.0%7D%7D%7D%7D)
## 🚀 Quick Start

### Installation

```bash
pip install transformers torch
```

### Inference Example

```python
from transformers import AutoProcessor, AutoModelForSequenceClassification

# Selects the task: "c2a", "dbo", or "vio"
task = "c2a"
model_id = f"abullard1/germeval2025-{task}-moderngbert-cw"

# Loads the model and processor
proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
mdl = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True).eval()

# Runs inference
text = "<your text>"
inputs = proc(text, return_tensors="pt", truncation=True)
probs = mdl(**inputs).logits.softmax(-1).detach().cpu().numpy()
print(probs)
```

## 🤖 Model Zoo

| Task | Class Weights | Class Weights + Focal Loss |
|------|---------------|----------------------------|
| **C2A** | [germeval2025-c2a-moderngbert-cw](https://huggingface.co/abullard1/germeval2025-c2a-moderngbert-cw) | [germeval2025-c2a-moderngbert-cw_and_focal](https://huggingface.co/abullard1/germeval2025-c2a-moderngbert-cw_and_focal) |
| **DBO** | [germeval2025-dbo-moderngbert-cw](https://huggingface.co/abullard1/germeval2025-dbo-moderngbert-cw) | [germeval2025-dbo-moderngbert-cw_and_focal](https://huggingface.co/abullard1/germeval2025-dbo-moderngbert-cw_and_focal) |
| **VIO** | [germeval2025-vio-moderngbert-cw](https://huggingface.co/abullard1/germeval2025-vio-moderngbert-cw) | [germeval2025-vio-moderngbert-cw_and_focal](https://huggingface.co/abullard1/germeval2025-vio-moderngbert-cw_and_focal) |

### 🎮 Interactive Demo
Try out the models in this interactive Huggingface Space:

<div align="center">

**[🔗 Live Demo: GermEval 2025 abullardUR Submission Models](https://huggingface.co/spaces/abullard1/abullardUR_GermEval2025_Submission_ModelZoo)**

*Experience our harmful content detection approach with this interactive demo. Use any of the provided example text snippets or use your own text to see how our models classify it.*

</div>

## 📋 Method

- **Architecture**: One model per subtask with mean-pooling head on ModernGBERT
- **Class Imbalance Handling**: 
  - Inverse-frequency class weights
  - Class weights + focal loss
- **Preprocessing**: Minimal preprocessing; retain anonymization tokens

### System Architecture

```mermaid
graph TD
    A[German Tweet Input] --> B[ModernGBERT Tokenizer]
    B --> C[ModernGBERT Encoder<br/>134M Parameters]
    C --> D[Mean Pooling Layer]
    D --> E[Classification Head]
    E --> F{Task Type}
    F -->|Binary| G[C2A: Call to Action<br/>VIO: Violence Detection]
    F -->|Multi-class| H[DBO: Democratic Basic Order<br/>4 Classes]
    G --> I[Sigmoid Output]
    H --> J[Softmax Output]
    
    style C fill:#e1f5fe
    style E fill:#f3e5f5
    style G fill:#e8f5e8
    style H fill:#fff3e0
```

### Training Pipeline

```mermaid
flowchart LR
    A[Raw Data<br/>CSV Files] --> B[Data Preprocessing<br/>Cleaning & Anonymization]
    B --> C[Train/Val Split<br/>80/20 Stratified]
    C --> D[Tokenization<br/>ModernGBERT Tokenizer]
    D --> E{Loss Function}
    E -->|Class Weights| F[Weighted Cross-Entropy<br/>Inverse Frequency]
    E -->|Focal Loss| G[Class Weights + Focal Loss<br/>γ Parameter Tuned]
    F --> H[Model Training<br/>W&B Hyperparameter Sweeps]
    G --> H
    H --> I[Best Model Selection<br/>Macro-F1 Score]
    I --> J[Final Evaluation<br/>Hidden Test Set]
    
    style A fill:#ffebee
    style E fill:#e3f2fd
    style I fill:#e8f5e8
    style J fill:#fff3e0
```

## 📁 Dataset

**Source**: GermEval 2025 shared task datasets (tweets from right-wing extremist networks, 2014–2016)

### Class Distribution Analysis

#### Dataset Statistics

| Subtask | Total Samples | Majority Class | Minority Class(es) | Imbalance Ratio |
|---------|---------------|----------------|-------------------|-----------------|
| **C2A** | 6,840 | False: 6,177 (90.3%) | True: 663 (9.7%) | 9.3:1 |
| **DBO** | 7,454 | Nothing: 6,277 (84.2%) | Subversive: 60 (0.8%) | 104.6:1 |
| **VIO** | 7,783 | False: 7,219 (92.8%) | True: 564 (7.2%) | 12.8:1 |

#### Class Imbalance Visualization

<div align="center">

| C2A (Call to Action) | DBO (Democratic Basic Order) | VIO (Violence Detection) |
|:---:|:---:|:---:|
| ![C2A Distribution](https://quickchart.io/chart?c=%7B%22type%22%3A%22doughnut%22%2C%22data%22%3A%7B%22labels%22%3A%5B%22False%22%2C%22True%22%5D%2C%22datasets%22%3A%5B%7B%22data%22%3A%5B90.3%2C9.7%5D%2C%22backgroundColor%22%3A%5B%22%23FF6384%22%2C%22%2336A2EB%22%5D%7D%5D%7D%2C%22options%22%3A%7B%22responsive%22%3Atrue%2C%22plugins%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22C2A%20Distribution%20%28%25%29%22%7D%7D%7D%7D) | ![DBO Distribution](https://quickchart.io/chart?c=%7B%22type%22%3A%22doughnut%22%2C%22data%22%3A%7B%22labels%22%3A%5B%22Nothing%22%2C%22Criticism%22%2C%22Agitation%22%2C%22Subversive%22%5D%2C%22datasets%22%3A%5B%7B%22data%22%3A%5B84.2%2C10.8%2C4.2%2C0.8%5D%2C%22backgroundColor%22%3A%5B%22%23FF6384%22%2C%22%2336A2EB%22%2C%22%23FFCE56%22%2C%22%234BC0C0%22%5D%7D%5D%7D%2C%22options%22%3A%7B%22responsive%22%3Atrue%2C%22plugins%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22DBO%20Distribution%20%28%25%29%22%7D%7D%7D%7D) | ![VIO Distribution](https://quickchart.io/chart?c=%7B%22type%22%3A%22doughnut%22%2C%22data%22%3A%7B%22labels%22%3A%5B%22False%22%2C%22True%22%5D%2C%22datasets%22%3A%5B%7B%22data%22%3A%5B92.8%2C7.2%5D%2C%22backgroundColor%22%3A%5B%22%23FF6384%22%2C%22%2336A2EB%22%5D%7D%5D%7D%2C%22options%22%3A%7B%22responsive%22%3Atrue%2C%22plugins%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22VIO%20Distribution%20%28%25%29%22%7D%7D%7D%7D) |
| **Binary**: 90.3% / 9.7% | **Multi-class**: 84.2% / 10.8% / 4.2% / 0.8% | **Binary**: 92.8% / 7.2% |

</div>

### Raw Data Files

<details>
<summary>Click to expand raw data structure</summary>

#### C2A (Call to Action)
- [c2a_train.csv](Competition-Solution/data/raw/c2a/c2a_train.csv)
- [c2a_trial.csv](Competition-Solution/data/raw/c2a/c2a_trial.csv)  
- [c2a_test.csv](Competition-Solution/data/raw/c2a/c2a_test.csv)

#### DBO (Democratic Basic Order)
- [dbo_train.csv](Competition-Solution/data/raw/dbo/dbo_train.csv)
- [dbo_trial.csv](Competition-Solution/data/raw/dbo/dbo_trial.csv)
- [dbo_test.csv](Competition-Solution/data/raw/dbo/dbo_test.csv)

#### VIO (Violence)
- [vio_train.csv](Competition-Solution/data/raw/vio/vio_train.csv)
- [vio_trial.csv](Competition-Solution/data/raw/vio/vio_trial.csv)
- [vio_test.csv](Competition-Solution/data/raw/vio/vio_test.csv)

</details>

### Processed Data Structure

- [processed/c2a/](Competition-Solution/data/processed/c2a/)
- [processed/dbo/](Competition-Solution/data/processed/dbo/)  
- [processed/vio/](Competition-Solution/data/processed/vio/)

Each processed dataset contains:
- `{task}_cleaned_trial.csv` - Cleaned trial dataset
- `{task}_cleaned_train.csv` - Cleaned training dataset
- `{task}_hf_dataset/` - Hugging Face dataset artifacts
- `{task}_hf_dataset_train/` - Training split HF dataset
- `{task}_hf_dataset_tokenized/` - Tokenized trial split
- `{task}_hf_dataset_tokenized_train/` - Tokenized training split

> **Note**: Trial splits were divided into train/validation/test based on the trial dataset. Training splits were divided into train/validation/test with the test split being the organizers' hidden test set for final evaluation.

>  The training dataset is also available on Huggingface 🤗 [here](https://huggingface.co/datasets/abullard1/germeval-2025-harmful-content-detection-training-dataset).

## ⚙️ Hyperparameters

### Final Model Configuration

| Subtask | Approach | Learning Rate | Weight Decay | Batch Size (Train/Eval) | Epochs | Warmup | Gamma | W&B Run |
|---------|----------|---------------|--------------|-------------------------|--------|--------|-------|---------|
| **C2A** | CW | 3e-5 | 0.0973 | 8/32 | 8 | 500 | - | [View Run](https://wandb.ai/uni-regensburg/Bachelors-Thesis/runs/f3sj4dsu) |
| **C2A** | FL | 5e-5 | 0.0227 | 16/16 | 5 | 100 | 0.599 | [View Run](https://wandb.ai/uni-regensburg/Bachelors-Thesis/runs/1rmk56s2) |
| **DBO** | CW | 1e-4 | 0.0417 | 8/16 | 5 | 500 | - | [View Run](https://wandb.ai/uni-regensburg/Bachelors-Thesis/runs/baojeybb) |
| **DBO** | FL | 3e-5 | 0.0270 | 16/16 | 8 | 100 | 1.537 | [View Run](https://wandb.ai/uni-regensburg/Bachelors-Thesis/runs/jztf6qdn) |
| **VIO** | CW | 5e-5 | 0.0301 | 16/32 | 3 | 300 | - | [View Run](https://wandb.ai/uni-regensburg/Bachelors-Thesis/runs/5nxgj5cu) |
| **VIO** | FL | 3e-5 | 0.0811 | 16/32 | 3 | 100 | 0.519 | [View Run](https://wandb.ai/uni-regensburg/Bachelors-Thesis/runs/atwbs5si) |


### Hyperparameter Optimization

All hyperparameters were optimized using Weights & Biases Bayesian sweeps:

| Subtask | Class-Weighted (CW) | Focal Loss (FL) |
|---------|-------------------|-----------------|
| **C2A** | [View Sweep](https://wandb.ai/uni-regensburg/Bachelors-Thesis/sweeps/x4cxix6l) | [View Sweep](https://wandb.ai/uni-regensburg/Bachelors-Thesis/sweeps/y6n4824y) |
| **DBO** | [View Sweep](https://wandb.ai/uni-regensburg/Bachelors-Thesis/sweeps/jeb5ysrp) | [View Sweep](https://wandb.ai/uni-regensburg/Bachelors-Thesis/sweeps/tttb1wf3) |
| **VIO** | [View Sweep](https://wandb.ai/uni-regensburg/Bachelors-Thesis/sweeps/5548o60h) | [View Sweep](https://wandb.ai/uni-regensburg/Bachelors-Thesis/sweeps/1adpvole) |

## 📁 Repository Structure

```
.
└── README.md           # this file
```

## 📄 Documentation

- [System Paper](System-Paper/abullardUR@GermEval-Shared-Task-2025.pdf)
- [Organizers' Overview Paper](https://www.aclweb.org/anthology/2025.germeval-1.1/)
- [ModernGBERT Paper](https://arxiv.org/abs/2505.13136) - Base model architecture and training details

## 📜 License

- **Code**: MIT
- **Model**: ModernGBERT base model: [Research-only RAIL-M license](https://huggingface.co/LSX-UniWue/ModernGBERT_134M/blob/main/license.md)
- **Fine-tuned models**: Inherit ModernGBERT's research-only license
- **Dataset**: [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html). Follow the license terms when using or redistributing the data.

## 📖 Citation (Preliminary)

```bibtex
@inproceedings{bullard2025germeval,
  title   = {abullardUR@GermEval Shared Task 2025: Fine-tuning ModernGBERT on Highly Imbalanced German Social Media for Harmful Content Detection},
  author  = {Bullard, Samuel},
  year    = {2025},
  booktitle = {Proceedings of KONVENS 2025 Workshops}
}
```

## 🙏 Acknowledgments

- Organizers of the GermEval 2025 shared task
- Prof. Dr. Udo Kruschwitz for supervision and guidance

---

<div align="center">

**[📊 Results](#-results) • [🚀 Quick Start](#-quick-start) • [🤖 Models](#-model-zoo) • [📁 Data](#-dataset) • [⚙️ Config](#️-hyperparameters)**

</div>
