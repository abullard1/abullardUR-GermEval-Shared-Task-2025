# Competition-Solution/src/evaluation_augment.py

"""
Evaluation Analysis Augmentation

This script extends the main evaluation stripts with four additional analyses based on
the results and discussion points that the main evaliation yielded.
We compute the following additional metrics:
    1. Expected Calibration Error (ECE) - So, do predicted probabilities match observed class frequencies?
    2. Error Overlap - Do the three loss strategies make the same or different mistakes?
    3. Confidence Distributions - How confident are models on correct vs. incorrect predictions?
    4. Vocabulary Overlap - Cross-corpus token analysis (GermEval 2025 vs. 2018 datasets) (only on C2A and VIO).

Author: Samuel Ruairí Bullard
Project: Evaluating Cost-Sensitive Loss Functions for Transformer-Based German Harmful Content Detection
Date: February 2026
"""

# Imports
import csv
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import jensenshannon
from torchmetrics.classification import MulticlassCalibrationError
from convokit import Corpus, Speaker, Utterance, FightingWords
from transformers import AutoTokenizer
from rich.console import Console
from rich.table import Table

# Local imports
from single_model_evaluation_utils import get_eval_config, load_model_and_data, predict, CLASS_NAMES

console = Console()

# Our nine fine-tuned competition models: 3 subtasks x 3 loss strategies
FINAL_COMP_MODELS = [
    ("C2A", "Baseline CE", "train-c2a-moderngbert-baseline-v1"),
    ("C2A", "CWCE", "train-c2a-moderngbert-v2-best-cw"),
    ("C2A", "CW+FL", "train-c2a-moderngbert-best-v4-best-fl"),
    ("DBO", "Baseline CE", "train-dbo-moderngbert-baseline-v1"),
    ("DBO", "CWCE", "train-dbo-moderngbert-v2-best-cw"),
    ("DBO", "CW+FL", "train-dbo-moderngbert-best-v3-best-fl"),
    ("VIO", "Baseline CE", "train-vio-moderngbert-baseline-v1"),
    ("VIO", "CWCE", "train-vio-moderngbert-v2-best-cw"),
    ("VIO", "CW+FL", "train-vio-moderngbert-best-v3-best-fl"),
]


# Loads a model, runs inference on its validation set, and frees the GPU memory.
# Returns predictions, softmax probabilities, and true labels.
def _load_and_predict(model_name, project_root):
    config = get_eval_config(model_name, str(project_root))

    model, processor, validation_dataset = load_model_and_data(config)

    predictions, probabilities, labels = predict(model, processor, validation_dataset, batch_size=32)

    # Deletes the model and processor from memory
    del model, processor

    # Checks if CUDA is available and clears the cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predictions, probabilities, labels


# --- Analysis 1: Expected Calibration Error (ECE) ---
# Naeini et al., 2015 https://doi.org/10.1609/aaai.v29i1.9602

def run_calibration_analysis(project_root):
    console.print("[bold]Analysis 1: Expected Calibration Error (ECE)[/bold]\n")

    # Creates a table to display the results
    table = Table(title="Calibration Summary")
    table.add_column("Task", style="cyan")
    table.add_column("Strategy", style="cyan")
    table.add_column("ECE", style="green")
    table.add_column("MCE", style="green")
    table.add_column("Overconfident %", style="yellow")

    # Empty dictionary to store the results
    results = {}

    # Loop through each model
    for subtask, strategy, model_name in FINAL_COMP_MODELS:
        console.print(f"Loading model: {model_name} ... ", end="")
        start_time = time.time()

        predictions, probabilities, true_labels = _load_and_predict(model_name, project_root)

        # ECE and MCE (equal-width binning, 15 bins)
        num_classes = probabilities.shape[1]
        probs_t = torch.tensor(probabilities, dtype=torch.float32)
        labels_t = torch.tensor(true_labels, dtype=torch.long)
        ece = float(MulticlassCalibrationError(num_classes, n_bins=15, norm="l1")(probs_t, labels_t))
        mce = float(MulticlassCalibrationError(num_classes, n_bins=15, norm="max")(probs_t, labels_t))

        # Checks for overconfidence, so how many high-confience predictions are actually wrong
        max_probs = np.max(probabilities, axis=1)
        high_confidence = max_probs > 0.9

        # The wrong presdictions
        wrong = predictions != true_labels

        # The number of high-confidence predictions
        n_high_confidence = int(high_confidence.sum())

        # The number of wrong predictions
        n_high_confidence_wrong = int((high_confidence & wrong).sum())

        # The overconfidence rate, e.g. the percentage of high-confidence predictions that are wrong
        overconfidence_rate = n_high_confidence_wrong / n_high_confidence if n_high_confidence > 0 else 0.0

        results[f"{subtask}_{strategy}"] = {
            "subtask": subtask, "strategy": strategy,
            "ece": ece, "mce": mce,
            "overconfidence_rate": float(overconfidence_rate),
            "n_samples": len(true_labels),
        }

        # Adds the results to the table
        table.add_row(subtask, strategy, f"{ece:.4f}", f"{mce:.4f}", f"{overconfidence_rate*100:.1f}")
        console.print(f"ECE={ece:.4f}  ({time.time()-start_time:.1f}s)")

    console.print(table)
    return results


# --- Analysis 2: Error Overlap ---

# Checks whether the three loss strategies make the same or different errors
def run_error_overlap_analysis(project_root):
    console.print("[bold]Analysis 2: Error Overlap Between Strategies[/bold]\n")

    # Empty dictionary to store the results
    results = {}

    # Loops through each subtask
    for subtask in ("C2A", "DBO", "VIO"):
        console.print(f"\n--- {subtask} ---")
        is_binary = subtask in ("C2A", "VIO")

        # Gets the three model names for the current subtask
        models = []
        for subtask_name, strategy, model_name in FINAL_COMP_MODELS:
            if subtask_name == subtask:
                models.append((strategy, model_name))

        # Loads all three and predicts on the same validation split
        all_preds = {}
        true_labels = None
        strategies = []

        # We loop thorugh the three loss strategies for the current subtask and get the predictions
        for strategy, model_name in models:
            console.print(f"Loading: {model_name} ... ", end="")
            start_time = time.time()

            # Loads the model and predicts on the validation split
            predictions, _, labels = _load_and_predict(model_name, project_root)
            all_preds[strategy] = predictions

            strategies.append(strategy)

            # Gets the true labels
            if true_labels is None:
                true_labels = labels
            
            console.print(f"done ({time.time()-start_time:.1f}s)")

        # Bool rror masks per strategy
        errors = {}

        # We create a boolean array for each strategy that indicates whether the prediction was wrong
        for strategy in strategies:
            errors[strategy] = all_preds[strategy] != true_labels

        strategy_0, strategy_1, strategy_2 = strategies

        # Threeway venn counts
        all_three = errors[strategy_0] & errors[strategy_1] & errors[strategy_2]
        errors_any = errors[strategy_0] | errors[strategy_1] | errors[strategy_2]
        total_errors_any = int(errors_any.sum())
        
        # Calculates the number of errors for each combination of strategies
        # Venn diagram of errors
        venn_diagram = {
            f"only_{strategy_0}": int((errors[strategy_0] & ~errors[strategy_1] & ~errors[strategy_2]).sum()),
            f"only_{strategy_1}": int((~errors[strategy_0] & errors[strategy_1] & ~errors[strategy_2]).sum()),
            f"only_{strategy_2}": int((~errors[strategy_0] & ~errors[strategy_1] & errors[strategy_2]).sum()),
            f"{strategy_0}+{strategy_1}_only": int((errors[strategy_0] & errors[strategy_1] & ~errors[strategy_2]).sum()),
            f"{strategy_0}+{strategy_2}_only": int((errors[strategy_0] & ~errors[strategy_1] & errors[strategy_2]).sum()),
            f"{strategy_1}+{strategy_2}_only": int((~errors[strategy_0] & errors[strategy_1] & errors[strategy_2]).sum()),
            "all_three": int(all_three.sum()),
        }

        # Pairwise Jaccard similarity calculation
        # pairwise jaccard tells us how many errors are shared between two strategies
        # It works by taking the intersection of errors and dividing by the union of errors
        pairwise = {}
        for index_i in range(3):
            for index_j in range(index_i + 1, 3):
                strategy_i, strategy_j = strategies[index_i], strategies[index_j]
                shared = int((errors[strategy_i] & errors[strategy_j]).sum())
                union = int((errors[strategy_i] | errors[strategy_j]).sum())
                pairwise[f"{strategy_i}_vs_{strategy_j}"] = {
                    "shared": shared,
                    "jaccard": shared / union if union > 0 else 0.0,
                }

        # The oracle tells us the best strategy per sample
        oracle_preds = np.copy(all_preds[strategy_0])

        # We loop through each sample and check if any strategy predicted the correct label
        # If so, we set the oracle prediction to that label
        for index in range(len(true_labels)):
            for strategy in strategies:
                if all_preds[strategy][index] == true_labels[index]:
                    oracle_preds[index] = true_labels[index]
                    break

        avg = "binary" if is_binary else "macro"
        oracle_f1 = float(f1_score(true_labels, oracle_preds, average=avg))

        individual_f1 = {}
        for strategy in strategies:
            individual_f1[strategy] = float(f1_score(true_labels, all_preds[strategy], average=avg))

        # We then do a per class breakdown to see what fraction of errors are shared by all three strategies
        class_names = CLASS_NAMES[subtask.lower()]
        per_class = {}

        # We loop through each class and calculate the number of shared and any errors
        for class_index in range(len(class_names)):
            class_mask = true_labels == class_index

            n_shared = int((all_three & class_mask).sum())
            n_any = int((errors_any & class_mask).sum())

            per_class[class_names[class_index]] = {
                "n": int(class_mask.sum()),
                "shared": n_shared, "any": n_any,
                "shared_rate": n_shared / n_any if n_any > 0 else 0.0,
            }

        # FP/FN for binary tasks
        fp_fn = None

        if is_binary:
            fp_fn = {}

            # We loop through each strategy and calculate the number of false positives and false negatives
            for strategy in strategies:
                fp_fn[strategy] = {
                    "FP": int(((all_preds[strategy] == 1) & (true_labels == 0)).sum()),
                    "FN": int(((all_preds[strategy] == 0) & (true_labels == 1)).sum()),
                }

        results[subtask] = {
            "n_samples": len(true_labels),
            "total_errors": {s: int(errors[s].sum()) for s in strategies},
            "total_errors_any": total_errors_any,
            "venn_diagram": venn_diagram, "pairwise": pairwise,
            "oracle_f1": oracle_f1,
            "individual_f1": individual_f1,
            "per_class_shared": per_class,
            "fp_fn": fp_fn,
        }

        shared_pct = venn_diagram['all_three'] / total_errors_any * 100

        console.print(f"Shared by all 3: {venn_diagram['all_three']}/{total_errors_any} ({shared_pct:.1f}%)")
        console.print(f"Oracle F1: {oracle_f1:.4f} (best individual: {max(individual_f1.values()):.4f})")

    return results


# --- Analysis 3: Confidence Distributions ---

# Profiles prediction confidence across the strategies to see if e.g. focal loss
# produces overconfident errors or CWCE shifts the distribution
def run_confidence_analysis(project_root):
    console.print("[bold]Analysis 3: Confidence Distributions[/bold]\n")

    table = Table(title="Confidence Summary")
    table.add_column("Task", style="cyan")
    table.add_column("Strategy", style="cyan")
    table.add_column("Mean Confidence", style="green")
    table.add_column("Conf Gap", style="green")
    table.add_column("Error Confidence", style="yellow")
    table.add_column("Error% > 0.9", style="red")

    results = {}
    for subtask, strategy, model_name in FINAL_COMP_MODELS:
        console.print(f"  Loading: {model_name} ... ", end="")
        start_time = time.time()

        predictions, probabilities, true_labels = _load_and_predict(model_name, project_root)
        max_probabilities = np.max(probabilities, axis=1)
        correct = predictions == true_labels
        wrong = ~correct

        mean_confidence = float(np.mean(max_probabilities))
        confidence_correct = float(np.mean(max_probabilities[correct])) if correct.any() else 0.0
        confidence_wrong = float(np.mean(max_probabilities[wrong])) if wrong.any() else 0.0
        confidence_gap = confidence_correct - confidence_wrong # How well the model "knows when it's wrong"

        # Error rate among predictions with confidence > 0.9
        above_09 = max_probabilities >= 0.9
        n_above = int(above_09.sum())
        n_wrong_above = int((above_09 & wrong).sum())
        err_rate_09 = n_wrong_above / n_above * 100 if n_above > 0 else 0.0

        # Per-class confidence
        # This is mostly interesting for the difficult DBO four-class problem
        class_names = CLASS_NAMES[subtask.lower()]
        per_class = {}
        for class_index in range(len(class_names)):
            class_mask = true_labels == class_index
            if class_mask.sum() == 0:
                continue
            class_probs = max_probabilities[class_mask]
            class_correct = correct[class_mask]
            per_class[class_names[class_index]] = {
                "n": int(class_mask.sum()),
                "accuracy": float(class_correct.mean()),
                "mean_confidence": float(np.mean(class_probs)),
                "confidence_correct": float(np.mean(class_probs[class_correct])) if class_correct.any() else 0.0,
                "confidence_wrong": float(np.mean(class_probs[~class_correct])) if (~class_correct).any() else 0.0,
            }

        results[f"{subtask}_{strategy}"] = {
            "subtask": subtask, "strategy": strategy,
            "n_samples": len(true_labels),
            "mean_confidence": mean_confidence,
            "confidence_correct": confidence_correct, "confidence_wrong": confidence_wrong,
            "confidence_gap": confidence_gap,
            "error_rate_above_09": err_rate_09,
            "per_class": per_class,
        }

        table.add_row(
            subtask, strategy,
            f"{mean_confidence:.3f}", f"{confidence_gap:.3f}", f"{confidence_wrong:.3f}", f"{err_rate_09:.1f}",
        )
        console.print(f"gap={confidence_gap:.3f}  ({time.time()-start_time:.1f}s)")

    console.print(table)
    return results


# --- Analysis 4: Cross-Corpus Vocabulary Overlap ---
# Reads the positive and negative texts from the datasets and counts the token frequencies
# We do this to see if there are any tokens that are distinctively more frequent in the positive class
def _load_labelled_texts(path, text_col, label_col, pos_value, delimiter=","):
    texts_positive = []
    texts_negative = []

    # Opens the dataset and reads the positive and negative texts
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            if str(row[label_col]).strip().upper() == str(pos_value).upper():
                texts_positive.append(row[text_col])
            else:
                texts_negative.append(row[text_col])
    return texts_positive, texts_negative


# Runs the model tokenizer on a list of texts and counts token frequencies.
def _tokenize_texts(tokenizer, texts):
    counter = Counter()
    for text in texts:
        counter.update(tokenizer.tokenize(text))
    return counter


# Fightin Words (Monroe et al., 2008)
# https://www.cambridge.org/core/journals/political-analysis/article/fightin-words-lexical-feature-selection-and-evaluation-for-identifying-the-content-of-political-conflict/81B3703230D21620B81EB6E2266C7A66
# Uses z-scored log-odds with an informative Dirichlet prior to find tokens
# that are distinctively more frequent in the positive class
def _fightin_words(texts_positive, texts_negative, tokenizer, top_k=100):
    # Builds the ConvoKit utterances from our labelled texts
    speaker = Speaker(id="anon")
    utterances = []
    for index, text in enumerate(texts_positive):
        utterances.append(Utterance(id=f"p{index}", text=text, speaker=speaker, meta={"cls": "positive"}))
    for index, text in enumerate(texts_negative):
        utterances.append(Utterance(id=f"n{index}", text=text, speaker=speaker, meta={"cls": "negative"}))

    # Passes our HF subword tokenizer so the analysis matches what the model sees
    count_vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize, lowercase=False)
    fighting_words = FightingWords(obj_type="utterance", cv=count_vectorizer)
    corpus = Corpus(utterances=utterances)

    # We fit the fighting words model to the corpus
    # This will calculate the z-scores for each ngram
    fighting_words.fit(corpus,
           class1_func=lambda u: u.meta["cls"] == "positive",
           class2_func=lambda u: u.meta["cls"] == "negative")

    # z-score DataFrame indexed by ngram, columns: "z-score", "class"
    # The zscore is the log-odds of the ngram being in the positive class
    # It effectively tells us how much more likely an ngram is to appear in the positive class
    # compared to the negative class
    zscores = fighting_words.get_ngram_zscores()

    # Gets the top k ngrams with the highest z-scores
    pos_df = zscores[zscores["class"] == "class1"].nlargest(top_k, "z-score")
    top_set = set(pos_df.index)

    # Gets the top 20 ngrams with the highest z-scores
    top_scored = []
    for ngram, score in pos_df["z-score"].head(20).items():
        top_scored.append((ngram, round(score, 3)))

    return top_set, top_scored


# Normalises a counter into a probability distribution over a fixed vocabulary.
def _counter_to_distribution(counter, vocabulary):
    counts_list = []

    # Iterates through the vocabulary and appends the count of each token to the list
    for token in vocabulary:
        counts_list.append(counter.get(token, 0))
    counts = np.array(counts_list, dtype=float)

    # Normalises the counts to a probability distribution
    total = counts.sum()

    # If the total is 0, returns the counts as is
    if total == 0:
        return counts
    return counts / total


# Compares the token distributions between two corpora (one subtask at a time).
# Returns the Jaccard overlap metrics and the Jensen-Shannon divergence.
def _analyze_corpus_pair(tokenizer, ge2025_positive, ge2025_negative, ge2018_positive, ge2018_negative):
    tokens_2025_positive = _tokenize_texts(tokenizer, ge2025_positive)
    tokens_2025_negative = _tokenize_texts(tokenizer, ge2025_negative)
    tokens_2018_positive = _tokenize_texts(tokenizer, ge2018_positive)
    tokens_2018_negative = _tokenize_texts(tokenizer, ge2018_negative)

    vocabulary_2025 = set((tokens_2025_positive + tokens_2025_negative).keys())
    vocabulary_2018 = set((tokens_2018_positive + tokens_2018_negative).keys())
    vocabulary_2025_positive = set(tokens_2025_positive.keys())
    vocabulary_2018_positive = set(tokens_2018_positive.keys())

    # Top-100 most distinctive positive-class tokens via FW (Monroe et al., 2008)
    top100_2025, top20_2025 = _fightin_words(ge2025_positive, ge2025_negative, tokenizer)
    top100_2018, top20_2018 = _fightin_words(ge2018_positive, ge2018_negative, tokenizer)

    # Jensen-Shannon Divergence (JSD) between the positive-class distributions
    shared_vocabulary_positive = sorted(vocabulary_2025_positive | vocabulary_2018_positive)

    # Converts the counters to probability distributions
    probability_distribution_2025_positive = _counter_to_distribution(tokens_2025_positive, shared_vocabulary_positive)
    probability_distribution_2018_positive = _counter_to_distribution(tokens_2018_positive, shared_vocabulary_positive)

    # Calculates the JSD
    jsd_positive_class = float(
        jensenshannon(probability_distribution_2025_positive, probability_distribution_2018_positive) ** 2
    )  # scipy package returns as sqrt(JSD)

    return {
        "corpus_stats": {
            "ge2025": {"n_positive": len(ge2025_positive), "n_negative": len(ge2025_negative), "vocab": len(vocabulary_2025)},
            "ge2018": {"n_positive": len(ge2018_positive), "n_negative": len(ge2018_negative), "vocab": len(vocabulary_2018)},
        },
        "overall_jaccard": len(vocabulary_2025 & vocabulary_2018) / len(vocabulary_2025 | vocabulary_2018),
        "pos_class_jaccard": len(vocabulary_2025_positive & vocabulary_2018_positive) / len(vocabulary_2025_positive | vocabulary_2018_positive),
        "distinctive_overlap": len(top100_2025 & top100_2018),
        "top100_in_target_pos": len(top100_2025 & vocabulary_2018_positive),
        "top100_in_target_any": len(top100_2025 & vocabulary_2018),
        "jsd_positive_class": jsd_positive_class,
        "shared_distinctive_tokens": sorted(list(top100_2025 & top100_2018)),
        "top20_distinctive_2025": top20_2025,
        "top20_distinctive_2018": top20_2018,
    }


# Runs the vocabulary overlap analysis for both C2A and VIO subtasks.
def run_vocabulary_overlap_analysis(project_root):
    console.print("[bold]Analysis 4: Cross-Corpus Vocabulary Overlap[/bold]\n")

    # All our competition models share the same base tokenizer (ModernGBERT)
    tokenizer_path = Path(project_root) / "models" / "finetuned_models" / "c2a" / "train-c2a-moderngbert-v2-best-cw" / "best_model_eval_f1-macro"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)

    # GermEval 2018 test set (used as target for both C2A and VIO snce these are the ones that have binary labels)
    ge2018_path = str(Path(project_root) / "data" / "evaluation" / "germeval2018" / "germeval2018.test.txt")
    ge2018_positive_texts, ge2018_negative_texts = _load_labelled_texts(ge2018_path, "text", "labels", "1")
    console.print(f"GE2018: {len(ge2018_positive_texts)} positive, {len(ge2018_negative_texts)} negative")

    # Builds a table to display the results
    table = Table(title="Vocabulary Overlap Summary")
    table.add_column("Subtask", style="cyan")
    table.add_column("Vocab Jaccard", style="green")
    table.add_column("Pos Jaccard", style="green")
    table.add_column("Distinctive Shared", style="yellow")
    table.add_column("JSD (pos)", style="red")

    # Empty dictionary to store the results
    results = {}

    # Only binary subtasks since DBO has no equivalent in GermEval 2018
    for subtask, label_col in [("C2A", "C2A"), ("VIO", "VIO")]:
        console.print(f"\n--- {subtask} ---")

        data_path = str(Path(project_root) / "data" / "raw" / subtask.lower() / f"{subtask.lower()}_train.csv")
        ge2025_positive_texts, ge2025_negative_texts = _load_labelled_texts(data_path, "description", label_col, "TRUE", delimiter=";")

        console.print(f"GE2025: {len(ge2025_positive_texts)} positive, {len(ge2025_negative_texts)} negative")

        # Analyzes the vocabulary overlap between the two corpora
        result = _analyze_corpus_pair(tokenizer, ge2025_positive_texts, ge2025_negative_texts, ge2018_positive_texts, ge2018_negative_texts)

        results[subtask] = result

        table.add_row(
            subtask,
            f"{result['overall_jaccard']:.3f}",
            f"{result['pos_class_jaccard']:.3f}",
            f"{result['distinctive_overlap']}/100",
            f"{result['jsd_positive_class']:.4f}",
        )

    console.print(table)
    return results


# --- Main ---
# Runs all four suplementary analyses and saves results to JSON files
def run_all_analyses(project_root, output_dir=None):
    # Checks if an output directory was provided, otherwise creates one
    if output_dir is None:
        output_dir = Path(project_root) / "analysis_exports" / "augmented"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    console.print(f"\n{'='*60}")
    console.print("[bold]Supplementary Evaluation Analyses[/bold]")
    console.print(f"{'='*60}\n")

    # Runs all four analyses
    all_results = {
        "calibration": run_calibration_analysis(project_root),
        "error_overlap": run_error_overlap_analysis(project_root),
        "confidence": run_confidence_analysis(project_root),
        "vocabulary_overlap": run_vocabulary_overlap_analysis(project_root),
    }

    # Loops through all the results and saves them to JSON files
    for name, data in all_results.items():
        with open(output_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    console.print(f"\nAll analyses complete in {time.time()-start_time:.1f}s")
    console.print(f"Results saved to: {output_dir}")

    return all_results

if __name__ == "__main__":
    project_root = str(Path(__file__).resolve().parent.parent)
    run_all_analyses(project_root)
