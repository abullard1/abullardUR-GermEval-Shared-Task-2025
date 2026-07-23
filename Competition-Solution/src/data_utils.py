# Competition-Solution/src/data_utils.py

"""
Data Processing and Analysis Utilities

This module provides the utilities for data processing, analysis,
and visualization It includes functions for dataset loading, text preprocessing,
statistical analysis, overlap detection, and data leakage prevention. In essence, 
it supports the jupyter notebooks and abstracts away much of the boilerplate 
code to make the notebooks more readable and easier to follow.

Author: Samuel Ruairí Bullard
Project: Evaluating Cost-Sensitive Loss Functions for Transformer-Based German Harmful Content Detection
Date: January 2026
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import os
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from rich.console import Console
from cleantext import clean
import validators
from collections import Counter
from rich.table import Table
from datasets import DatasetDict

console = Console()


# Loads the competition data from the specified directory (c2a, dbo, vio)
def load_competition_data(data_dir, type):
    console.print("Loading competition data...", style="italic")
    
    # Validates the input parameters
    if data_dir is None or type is None:
        console.print("Error: Both data directory and type must be specified.", style="bold red")
        return None, None, None

    if not os.path.exists(data_dir):
        console.print(f"Error: Directory {data_dir} does not exist.", style="bold red")
        return None, None, None

    # Constructs the file paths for each dataset
    data_c2a = data_dir + "c2a" + "/" + "c2a_" + type + ".csv"
    data_dbo = data_dir + "dbo" + "/" + "dbo_" + type + ".csv" 
    data_vio = data_dir + "vio" + "/" + "vio_" + type + ".csv"

    try:
        # Loads the data using ";" as the delimiter
        df_call2action = pd.read_csv(data_c2a, sep=";")
        df_fdgo = pd.read_csv(data_dbo, sep=";")
        df_violence = pd.read_csv(data_vio, sep=";")

        # Assigns a name to each dataframe
        df_call2action.name = "Call2Action"
        df_fdgo.name = "DBO"
        df_violence.name = "Violence"
        
        console.print(f"Data of type \"{type}\" loaded successfully.", style="bold green")
        return df_call2action, df_fdgo, df_violence

    except FileNotFoundError as e:
        console.print(f"Error loading the data: {e}", style="bold red")
        return None, None, None
    

# Saves a dataframe to a csv file to a specified directory
def save_dataframe_to_csv(dataframe, directory, filename, suffix=""):
    # Verifies the input parameters
    if dataframe is None:
        console.print("Error: No dataframe provided.", style="bold red")
        return
    if directory is None:
        console.print("Error: No directory provided.", style="bold red")
        return
    if filename is None:
        console.print("Error: No filename provided.", style="bold red")
        return

    # Gets the name of the dataframe
    df_has_name = hasattr(dataframe, "name") and dataframe.name is not None
    df_name = filename if filename else (dataframe.name if df_has_name else "unnamed")

    # Constructs the final filename with a suffix if provided
    suffix_text = f"_{suffix}" if suffix else ""
    final_filename = f"{filename}{suffix_text}.csv"

    # Saves the dataframe to a csv file
    console.print(f"Saving dataframe \"{df_name}\" to {directory}/{final_filename}...", style="italic")
    if os.path.exists(f"{directory}/{final_filename}"):
        console.print(f"File \"{final_filename}\" already exists. Overwriting...", style="yellow")
    dataframe.to_csv(f"{directory}/{final_filename}", index=False, sep=";")
    console.print(f"Dataframe \"{df_name}\" saved successfully.", style="bold green")

# Saves a dictionary of dataframes to a csv file to a specified directory and subfolder
def save_dataframes_to_csv(dataframes_dict, base_directory, suffix=""):
    # Verifies the input parameters
    if dataframes_dict is None:
        console.print("Error: No dataframes provided.", style="bold red")
        return
    if base_directory is None:
        console.print("Error: No base directory provided.", style="bold red")
        return
    
    # Saves the dataframes to csv files using the save_dataframe_to_csv function
    for filename, (dataframe, subfolder) in dataframes_dict.items():
        # Creates the full directory path including the subfolder
        full_directory = os.path.join(base_directory, subfolder)
        # Creates the directory if it doesn't exist
        os.makedirs(full_directory, exist_ok=True)
        # Saves the dataframe to the csv file
        save_dataframe_to_csv(dataframe, full_directory, filename, suffix)

    console.print("All dataframes saved successfully.", style="bold green")


# Initializes NLTK by downloading the required resources
def initialize_nltk(quiet=True):
    console.print(f"Initializing NLTK...\nResources will be downloaded to {nltk.data.path[0]}.", style="italic")

    # Downloads the tokenizer and stopwords
    resources = [
        "punkt_tab",
        "stopwords"
    ]
    
    # Loops through the NLTK resources and downloads them if not already downloaded
    for resource in resources:
        console.print(f"Downloading {resource}...", style="italic")
        try:
            if not nltk.download(resource, quiet=quiet):
                console.print(f"Error: {resource} download failed.", style="bold red")
                return
        except Exception as e:
            console.print(f"Error downloading {resource}: {e}", style="bold red")
            return

    console.print("NLTK initialized successfully.", style="bold green")

    
# Preprocesses a text string for frequency analysis
def prepare_text_for_frequency_distribution_analysis(text, language="german"):
    console.print("Preparing text string for frequency distribution analysis...", style="italic")

    # Validates the input parameters
    if text is None or not isinstance(text, str):
        console.print("Error: No valid text string provided.", style="bold red")
        return None, None
    
    try:
        # Tokenizes the tweets into words and sentences
        # Uses an improved TreebankWordTokenizer along with PunktSentenceTokenizer
        # Sources TreebankWordTokenizer: https://www.nltk.org/api/nltk.tokenize.treebank.html (Papers in to read library in Papers by ReadCube)
        # Sources PunktSentenceTokenizer: https://www.nltk.org/api/nltk.tokenize.punkt.html (Papers in to read library in Papers by ReadCube)
        tokens = word_tokenize(text=text.lower(), language=language, preserve_line=False)
        sentences = sent_tokenize(text=text.lower(), language=language)

        console.print("Text prepared successfully.", style="bold green")
        return tokens, sentences

    except Exception as e:
        console.print(f"Error preparing the text: {e}", style="bold red")
        return None, None
    
# Preprocesses text data for frequency analysis
# (Applied to a subtask dataset and uses the above defined per description text processing function)
def prepare_data_for_frequency_distribution_analysis(data, language="german", text_column="description"):
    console.print("Preparing data for frequency distribution analysis...", style="italic")

    if data is None:
        console.print("Error: No data provided.", style="bold red")
        return None, None

    # Combines all tweets into a single text
    all_tweets_text = " ".join(data[text_column])
    
    # Uses the text processing function
    return prepare_text_for_frequency_distribution_analysis(all_tweets_text, language)

    
# Removes stopwords from a token list
def get_filtered_tokens(tokens):
    german_stopwords = stopwords.words("german")
    filtered_words = []
    
    for word in tokens:
        if word.isalpha() and word not in german_stopwords:
            filtered_words.append(word)
            
    return filtered_words


# Takes a dataframe and returns a dictionary of class corpuses
# e.g. {C2A_true: "C2A_true_corpus", C2A_false: "C2A_false_corpus"}
# e.g. {DBO_nothing: "DBO_nothing_corpus", DBO_criticism: "DBO_criticism_corpus", DBO_subversive: "DBO_subversive_corpus", DBO_agitation: "DBO_agitation_corpus"}
# e.g. {VIO_true: "VIO_true_corpus", VIO_false: "VIO_false_corpus"}
def get_class_corpuses(dataframe, label_column, text_column="description"):
    console.print(f"Creating class corpuses for {label_column} column...", style="italic")

    # Validates the input parameters
    if dataframe is None or dataframe.empty:
        console.print("Error: Empty dataframe provided.", style="bold red")
        return {}
    if label_column is None:
        console.print("Error: Label column must be specified.", style="bold red")
        return {}
    if text_column is None:
        console.print("Error: Text column must be specified.", style="bold red")
        return {}
    if text_column not in dataframe.columns:
        console.print(f"Error: Text column \"{text_column}\" not found in dataframe.", style="bold red")
        return {}
    if label_column not in dataframe.columns:
        console.print(f"Error: Label column \"{label_column}\" not found in dataframe.", style="bold red")
        return {}
    
    # Dictionary to store the class corpuses
    class_corpuses = {}
    
    # Gets all unique values in the label column
    unique_labels = dataframe[label_column].unique()
    
    # Checks if there are any unique labels
    if len(unique_labels) == 0:
        console.print(f"Error: No unique labels found in \"{label_column}\" column.", style="bold red")
        return {}
    
    # Iterates over each unique label
    for label in unique_labels:
        # Filters the dataframe for the current label
        class_df = dataframe[dataframe[label_column] == label]
        
        # Checks if there are any tweets for the current label
        if not class_df.empty:
            # Concatenates all tweet texts for this class
            class_corpus = " ".join(class_df[text_column])
            
            # Creates a key for the class corpus: {LabelColumn}_{LabelValue}
            label_str = str(label).lower()
            key = f"{label_column}_{label_str}"
            
            # Adds the class corpus to the dictionary
            class_corpuses[key] = class_corpus
            
            console.print(f"  - Found {len(class_df)} tweets for label \"{label}\"", style="blue")
        else:
            console.print(f"Warning: No data found for label \"{label}\".", style="yellow")
    
    # Completion message
    if class_corpuses:
        console.print(f"Created {len(class_corpuses)} class corpuses successfully.", style="bold green")
        for key in class_corpuses.keys():
            # Gets the word count of the corpus
            word_count = len(class_corpuses[key].split())
            # Prints the word count of the corpus
            console.print(f"  - {key}: ~{word_count} words", style="green")
    else:
        console.print("No class corpuses were created.", style="yellow")

    return class_corpuses


# Calculates dataset overlaps
# Calculating these overlaps is crucial for the data leakage prevention step
def calculate_dataset_overlaps(datasets, column="description", show_details=True):
    # Validates the input parameters
    if not datasets or len(datasets) < 2:
        console.print("Error: At least two datasets are required to calculate overlap.", style="bold red")
        return {"pairwise": {}, "total": set()}
    
    # Extracts names and dataframes
    dataset_names = []
    for name, _ in datasets:
        dataset_names.append(name)
    joined_dataset_names = ", ".join(dataset_names)
    console.print(f"Calculating overlaps across {len(datasets)} datasets: {joined_dataset_names}", style="bold cyan")
    
    # Stores per-pair overlap sets (pair_key -> set of items present in both datasets)
    pairwise_overlaps = {}

    # Tracks all items that appear in at least two datasets (union of per-pair overlaps)
    all_overlap_items = set()
    
    # Calculates pairwise overlaps

    # Initializes pair index for display numbering
    i = -1
    # Loops through the datasets
    for idx1 in range(len(datasets)):
        # Gets the name and dataframe of the current dataset
        name1, df1 = datasets[idx1]

        # Loops through the remaining datasets
        for idx2 in range(idx1 + 1, len(datasets)):
            name2, df2 = datasets[idx2]

            # Increments the pair index
            i += 1

            # Creates a key for the current pair
            pair_key = f"{name1}_{name2}"
            
            # Ensures both dataframes exists
            if df1 is None or df2 is None:
                console.print(f"Skipping {pair_key} as one or both dataframes are None.", style="yellow")
                pairwise_overlaps[pair_key] = set()
                continue
            
            # Displays details if show_details is True
            if show_details:
                console.print(f"\n--- Pair {i+1}: {name1} vs {name2} ---", style="italic blue")
            
            # Gets unique non-null values from both datasets
            values1 = set(df1[column].dropna().astype(str).unique())
            values2 = set(df2[column].dropna().astype(str).unique())
            
            # Calculates intersection and percentage
            overlap_set = values1.intersection(values2)
            if values1:  # Avoids division by zero
                overlap_percent1 = len(overlap_set) / len(values1) * 100
            else:
                overlap_percent1 = 0
                
            if values2:  # Avoids division by zero
                overlap_percent2 = len(overlap_set) / len(values2) * 100
            else:
                overlap_percent2 = 0
            
            # Stores the results
            pairwise_overlaps[pair_key] = overlap_set
            all_overlap_items.update(overlap_set)
            
            # Displays details if show_details is True
            if show_details:
                console.print(f"{len(overlap_set)} {column}s appear in both datasets:", style="green")
                console.print(f"- {len(overlap_set)}/{len(values1)} from {name1} ({overlap_percent1:.2f}%)")
                console.print(f"- {len(overlap_set)}/{len(values2)} from {name2} ({overlap_percent2:.2f}%)")
    
    # Displays the summary of the total overlap
    console.print(f"\nTotal unique {column}s appearing in at least two datasets: [b]{len(all_overlap_items)}[/b]", style="bold magenta")
    
    # Returns both the pairwise and total results
    return {
        "pairwise": pairwise_overlaps,
        "total": all_overlap_items
    }


# Displays the overlap percentages for the datasets
def display_dataset_overlap_percentages(overlaps, datasets, column="description", show_pairwise=True):
    # Validates the input parameters
    if not overlaps or "total" not in overlaps or "pairwise" not in overlaps:
        console.print("Error: Invalid overlaps data provided.", style="bold red")
        return {}
    
    # Extracts total overlap and dataset information
    total_overlap = overlaps["total"]
    pairwise_overlaps = overlaps["pairwise"]
    
    # Computes the union of unique items across datasets (denominator for rates)
    all_unique_values = set()
    dataset_values = {}
    
    # Iterates over datasets and collects the unique non-null values for the specified column  
    for name, dataset_dataframe in datasets:
        if dataset_dataframe is not None and column in dataset_dataframe.columns:
            unique_values = set(dataset_dataframe[column].unique())
            all_unique_values.update(unique_values)
            dataset_values[name] = unique_values
    
    # Computes the overall overlap percentage (items in 2+ datasets / all unique items)
    total_overlap_percent = len(total_overlap) / len(all_unique_values) * 100 if all_unique_values else 0
    
    # Initializes the results dictionary
    results = {
        "total_percent": total_overlap_percent,
        "unique_items_count": len(all_unique_values),
        "overlap_items_count": len(total_overlap),
        "pairwise_percentages": {}
    }
    
    console.rule(f"[bold cyan]Overlap Percentages for \"{column}\" across {len(datasets)} datasets[/bold cyan]")
    
    # Displays the overall overlap summary
    console.print(f"Total overlap: [bold]{len(total_overlap)}/{len(all_unique_values)}[/bold] unique {column}s appear in 2+ datasets ([bold]{total_overlap_percent:.2f}%[/bold])", style="magenta")
    
    # Calculates and displays the pairwise percentages if show_pairwise is True
    if show_pairwise and pairwise_overlaps:
        console.print("\nPairwise percentages:", style="cyan")
        
        # Loops through the pairwise overlaps
        for pair_key, overlap_set in pairwise_overlaps.items():
            # Splits the pair key to get the dataset names
            names = pair_key.split("_")

            # Checks if the dataset names are valid
            if len(names) == 2 and names[0] in dataset_values and names[1] in dataset_values:
                name1, name2 = names
                values1 = dataset_values[name1]
                values2 = dataset_values[name2]
                
                # Computes the per-dataset relative overlap percentages
                percent1 = len(overlap_set) / len(values1) * 100 if values1 else 0
                percent2 = len(overlap_set) / len(values2) * 100 if values2 else 0
                
                # Computes the percentage relative to the union of both datasets
                combined = len(values1.union(values2))
                percent_combined = len(overlap_set) / combined * 100 if combined else 0
                
                # Stores the percentages in the results dictionary
                results["pairwise_percentages"][pair_key] = {
                    "percent_first": percent1,
                    "percent_second": percent2,
                    "percent_combined": percent_combined,
                    "overlap_count": len(overlap_set),
                    "first_count": len(values1),
                    "second_count": len(values2),
                    "combined_count": combined
                }
                
                # Displays the percentages
                console.print(f"\n[bold]{name1}[/bold] and [bold]{name2}[/bold] share [bold]{len(overlap_set)}[/bold] {column}s:", style="green")
                console.print(f"- {len(overlap_set)}/{len(values1)} from {name1} ([bold]{percent1:.2f}%[/bold])")
                console.print(f"- {len(overlap_set)}/{len(values2)} from {name2} ([bold]{percent2:.2f}%[/bold])")
                console.print(f"- {len(overlap_set)}/{combined} of all unique {column}s between them ([bold]{percent_combined:.2f}%[/bold])")
    
    console.line()

    return results


# Helper function to display tweet details for get_shortest_longest_tweets below
def display_selected_tweets(selected_tweets_dataframe, text_column_name, selection_title, filter_min_len, filter_max_len, num_to_display):
    console.print(f"\n--- Displaying up to {num_to_display} {selection_title} Tweets (Lengths from {int(filter_min_len)} to {int(filter_max_len)}) ---", style="bold cyan")

    # Checks if the dataframe is empty
    if selected_tweets_dataframe.empty:
        console.print(f"No tweets found in the specified {selection_title.lower()} range.", style="yellow")

    # Displays the tweets if the dataframe is not empty
    else:
        # Iterates over the dataframe
        for i, row in selected_tweets_dataframe.iterrows():
            # Gets the text and length of the tweet
            text = row[text_column_name]
            length = row["length"]

            # Gets original index display value (if available)
            if "index" in selected_tweets_dataframe.columns:
                display_index = str(row["index"])
            else:
                display_index = str(i)
                
            console.print(f"Index: {display_index}, Length: {int(length)}")
            console.print(f"Text: \"{text}\"")

            # Line break between tweets
            if i < len(selected_tweets_dataframe) - 1:
                console.line()


# Gets the shortest and longest tweets from a dataframe
def get_shortest_longest_tweets(dataframe, column="description", top_n=3, short_length_range=1, long_length_range=1):
    console.print(f"Getting up to {top_n} tweets from specified length ranges (Short range span: {short_length_range}, Long range span: {long_length_range})...", style="italic")

    # Validates the input parameters
    if dataframe is None or dataframe.empty:
        console.print("Error: No or empty dataframe provided.", style="bold red")
        return pd.Series(dtype="object"), pd.Series(dtype="object")
    if column not in dataframe.columns:
        console.print(f"Error: Column \"{column}\" not found in dataframe.", style="bold red")
        return pd.Series(dtype="object"), pd.Series(dtype="object")

    # Creates a copy of the dataframe
    df_with_len = dataframe.copy()

    # Converts the text column to a string
    df_with_len[column] = df_with_len[column].astype(str)

    # Adds a length column
    df_with_len["length"] = df_with_len[column].str.len()

    # Checks if the dataframe is empty, returns an empty series if so (No tweets found in the specified length ranges)
    if df_with_len.empty or df_with_len["length"].isnull().all():
        console.print("Dataframe is effectively empty or lengths could not be calculated.", style="yellow")
        return pd.Series(dtype="object"), pd.Series(dtype="object")

    # Gets the minimum and maximum lengths
    min_actual_length = df_with_len["length"].min()
    max_actual_length = df_with_len["length"].max()

    # Defines the target length ranges for filtering
    filter_short_min_len = min_actual_length
    filter_short_max_len = min_actual_length + short_length_range - 1
    
    filter_long_min_len = max_actual_length - long_length_range + 1
    filter_long_max_len = max_actual_length

    # Resets the index to avoid index-related sorting issues
    df_with_len = df_with_len.reset_index()
    
    # Filters for shortest tweets
    shortest_tweets_df = df_with_len[
        (df_with_len["length"] >= filter_short_min_len) &
        (df_with_len["length"] <= filter_short_max_len)
    ]
    
    # Sorts shortest tweets by length (ascending)
    if not shortest_tweets_df.empty:
        shortest_tweets_df = shortest_tweets_df.sort_values(by="length", ascending=True).head(top_n)
    
    # Filters for the longest tweets
    longest_tweets_df = df_with_len[
        (df_with_len["length"] >= filter_long_min_len) &
        (df_with_len["length"] <= filter_long_max_len)
    ]
    
    # Sorts the longest tweets by length (descending)
    if not longest_tweets_df.empty:
        longest_tweets_df = longest_tweets_df.sort_values(by="length", ascending=False).head(top_n)
    
    # Displays the results using the helper function defined above
    display_selected_tweets(shortest_tweets_df, column, "Shortest", filter_short_min_len, filter_short_max_len, top_n)
    display_selected_tweets(longest_tweets_df, column, "Longest", filter_long_min_len, filter_long_max_len, top_n)
            
    console.print("\nFinished getting tweets from specified length ranges.", style="italic")
    
    # Returns the shortest and longest tweets
    return_shortest = shortest_tweets_df[column] if not shortest_tweets_df.empty else pd.Series(dtype="object")
    return_longest = longest_tweets_df[column] if not longest_tweets_df.empty else pd.Series(dtype="object")

    # Returns the shortest and longest tweets
    return return_shortest, return_longest


# Extracts n-grams from a token list
def generate_ngrams_from_tokens(tokens, n=2):
    console.print(f"Generating {n}-grams from tokens...", style="italic")

    # Validates the input parameters
    if tokens is None or len(tokens) < n:
        console.print(f"Error: Not enough tokens to generate {n}-grams.", style="bold red")
        return None
    
    try:
        ngram_list = []
        
        # Creates n-grams from the token list
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngram_list.append(ngram)
        
        # Creates a frequency distribution of n-grams
        freq_dist = FreqDist(ngram_list)

        console.print(f"Generated {len(ngram_list)} {n}-grams successfully.", style="bold green")
        return freq_dist

    except Exception as e:
        console.print(f"Error generating {n}-grams: {e}", style="bold red")
        return None


# Cleans the text in a dataframe column (URLs and emails are replaced with placeholder tokens)
# Hint: Passing None to the url_replacement or email_replacement will effectively disable the replacement (As per the cleantext library's implementation)
def clean_text(dataframe, column_name="description", url_replacement="[URL]", email_replacement="[EMAIL]"):
    if column_name not in dataframe.columns:
        console.print(f"Error: Column \"{column_name}\" not found in dataframe.", style="bold red")
        return dataframe

    df_copy = dataframe.copy()
    
    # Fills empty ("na") values with an empty string and converts the entries in the column to strings
    texts_to_process = df_copy[column_name].fillna("").astype(str)

    # Cleans the text in the specified column
    cleaned_column_data = []
    for text in texts_to_process:
        cleaned_text = clean(
            text,
            lang="de",
            replace_with_url=url_replacement,
            replace_with_email=email_replacement,
            
            # Essentially ON/OFF switch for replacing the URLs/E-Mails
            no_urls=True,
            no_emails=True,
            
            # Disables the default transformations so that we only affect URLs/E-Mails
            lower=False,
            fix_unicode=False,
            to_ascii=False,
            normalize_whitespace=False,
            strip_lines=False,
            no_line_breaks=False
        )
        cleaned_column_data.append(cleaned_text)
    
    # Replaces the text in the specified column with the cleaned text
    df_copy[column_name] = cleaned_column_data
    return df_copy


# Checks and counts a pandas dataframe for URL occurances
def check_for_urls(dataframe, column_name="description", show_random_hit_examples=False):
    if column_name not in dataframe.columns:
        console.print(f"Error: Column \"{column_name}\" not found in dataframe.", style="bold red")
        return dataframe
    
    # Checks if the dataframe has the name attribute and if it is not None
    df_has_name = hasattr(dataframe, "name") and dataframe.name is not None
    
    url_count = 0
    df_hits = pd.DataFrame(columns=[column_name])

    # Uses the validators library to check for URLs in the specified column
    for tweet in dataframe[column_name]:
        if validators.url(tweet):
            url_count += 1
            df_hits.loc[len(df_hits)] = [tweet]

    # Displays random examples if show_random_hit_examples is True
    if show_random_hit_examples and not df_hits.empty:
        np.random.seed(42)

        sample_size = min(3, len(df_hits))

        # Samples the dataframe
        samples = df_hits.sample(n=sample_size)

        console.print("\nRandom examples containing URLs:", style="cyan")

        # Loops through the samples and displays the random examples
        for _, row in samples.iterrows():
            console.print(f"- {row[column_name]}")

    dataframe_name = dataframe.name if df_has_name else "unnamed"
    if url_count > 0:
        console.print(f"Total URLs found in the \"{column_name}\" column of dataframe \"{dataframe_name}\": {url_count}", style="bold green")
    else:
        console.print(f"No URLs found in the \"{column_name}\" column of dataframe \"{dataframe_name}\".", style="bold red")
    
    return url_count


# Checks and counts a pandas dataframe for email occurances
def check_for_emails(dataframe, column_name="description", show_random_hit_examples=False):
    if column_name not in dataframe.columns:
        console.print(f"Error: Column \"{column_name}\" not found in dataframe.", style="bold red")
        return dataframe
    
    df_has_name = hasattr(dataframe, "name") and dataframe.name is not None
    
    email_count = 0
    df_hits = pd.DataFrame(columns=[column_name])

    # Uses the validators library to check for emails in the specified column
    for tweet in dataframe[column_name]:
        if validators.email(tweet):
            email_count += 1
            df_hits.loc[len(df_hits)] = [tweet]

    # Displays random examples if show_random_hit_examples is True
    if show_random_hit_examples and not df_hits.empty:
        np.random.seed(42)

        # Specifies the sample size
        sample_size = min(3, len(df_hits))

        # Samples the dataframe
        samples = df_hits.sample(n=sample_size)

        console.print("\nRandom examples containing emails:", style="cyan")

        # Loops through the samples and displays the random examples
        for _, row in samples.iterrows():
            console.print(f"- {row[column_name]}")

    dataframe_name = dataframe.name if df_has_name else "unnamed"
    if email_count > 0:
        console.print(f"Total emails found in the \"{column_name}\" column of dataframe \"{dataframe_name}\": {email_count}", style="bold green")
    else:
        console.print(f"No emails found in the \"{column_name}\" column of dataframe \"{dataframe_name}\".", style="bold red")
    
    return email_count


# Compares the original dataset class distribution with the stratified splits
def show_hf_stratification_results(original_dataset, split_dataset, label_column):
    if not original_dataset or not split_dataset:
        console.print("Error: Invalid dataset(s) provided.", style="bold red")
        return
    
    # Gets the original distribution
    original_counter = Counter(original_dataset[label_column])
    original_total = len(original_dataset)
    
    # Gets the split distributions
    # Note: We exclude the test split since the test set we use to make our
    # model submissions does not have any labels
    splits = []

    # Loops through the splits and adds them to the list if they are not the test split
    for split in split_dataset.keys():
        if split != "test":
            splits.append(split)

    # Initializes the counters and totals for each split
    split_counters = {}
    split_totals = {}
    
    # Calculates the counters and totals for each split by looping through the splits
    for split in splits:
        split_counters[split] = Counter(split_dataset[split][label_column])
        split_totals[split] = len(split_dataset[split])
    
    console.print(f"\n[bold cyan]{label_column} Stratification Verification[/bold cyan]")
    
    # Displays the dataset sizes
    sizes = [f"Original: {original_total}"]

    # Adds the split sizes and percentages
    for split in splits:
        split_size = split_totals[split]
        split_pct = split_size / original_total * 100
        sizes.append(f"{split.title()}: {split_size} ({split_pct:.1f}%)")
    
    # Adds the test split size for reference
    if "test" in split_dataset:
        test_size = len(split_dataset["test"])
        sizes.append(f"Test: {test_size} (separate dataset)")
    
    console.print(" | ".join(sizes) + "\n")
    
    # Creates the comparison table
    table = Table()
    table.add_column("Class", style="cyan")
    table.add_column("Original%", justify="right")

    # Adds a column for each split
    for split in splits:
        table.add_column(f"{split.title()}%", justify="right")
    table.add_column("Max Diff", justify="right")
    
    # Adds a row for each class
    for label in sorted(original_counter.keys()):
        original_percentage = original_counter[label] / original_total * 100
        split_percentages = []
        for split in splits:
            count = split_counters[split].get(label, 0)
            total = split_totals[split]
            split_percentage = count / total * 100
            split_percentages.append(split_percentage)
        
        # Calculates the maximum difference from the original
        max_diff = 0
        for split_percentage in split_percentages:
            percentage_difference = abs(split_percentage - original_percentage)
            if percentage_difference > max_diff:
                max_diff = percentage_difference
        
        # Colors the difference
        if max_diff < 2.0:
            diff_color = "green"
        elif max_diff < 5.0:
            diff_color = "yellow"
        else:
            diff_color = "red"
        
        # Prepares the row data
        row = [str(label), f"{original_percentage:.1f}%"]

        # Loops through the split percentages and adds them to the row
        for split_percentage in split_percentages:
            row.append(f"{split_percentage:.1f}%")
        row.append(f"[{diff_color}]{max_diff:.1f}%[/{diff_color}]")
        
        # Adds the row to the table
        table.add_row(*row)
    
    console.print(table)


# Removes duplicate tweet descriptions from a dataset's validation set as not to leak information to the model, thereby causing a bias in the model's performance.
# Hint 1: Only used for the training dataset, since the trial datset had barely and overlapping tweet descriptions.
# Hint 2: This targets only the training and validation set, since the test set is provided by the competition organizers and not split by us.
def handle_dataset_leakage(dataset, text_col="description"):
    # Gets the unique tweet descriptions from the training set
    train_desc = set(dataset["train"][text_col])

    # Gets the unique tweet descriptions from the validation set
    val_desc = dataset["validation"][text_col]
    original_val_size = len(val_desc)

    # Keeps only the indices of tweet descriptions that are not in the training set
    keep_indices = []

    # Loops through the validation set and adds the indices of the tweet descriptions that are not in the training set to the keep_indices list
    for i, d in enumerate(val_desc):
        if d not in train_desc:
            keep_indices.append(i)
    
    # Calculates the number of dropped samples
    dropped_count = original_val_size - len(keep_indices)
    console.print(f"Removed [red]{dropped_count}[/red] duplicate samples from validation set ([cyan]{original_val_size}[/cyan] -> [green]{len(keep_indices)}[/green])")
    
    # Returns the dataset dict with the validation set filtered (i.e. without the duplicate tweet descriptions)
    return DatasetDict({
        "train": dataset["train"],
        "validation": dataset["validation"].select(keep_indices),
        "test": dataset["test"],
    })


# --- Plotting Functions ---
# Creates a plot for the n-gram frequency distributions
def create_ngram_frequency_plot(ngram_freq_dict, title="N-gram Frequency Distribution", n_items=15, figsize=(16, 10), show_grid=False):
    console.print("Creating n-gram frequency distribution plot...", style="italic")
    
    # Validates the input parameters
    if not ngram_freq_dict:
        console.print("Error: No n-gram frequency distributions provided.", style="bold red")
        return None, None
    
    # Converts the n-gram tuples to strings for plotting
    plot_ready_dict = {}
    for label, freq_dist in ngram_freq_dict.items():
        # Creates a new frequency distribution with string keys
        converted_dist = FreqDist()

        # Loops through the n-gram tuples and adds them to the frequency distribution
        for ngram_tuple, freq in freq_dist.items():
            ngram_str = " ".join(ngram_tuple)
            converted_dist[ngram_str] = freq
        
        plot_ready_dict[label] = converted_dist

    # Calculates the grid size based on the number of plots
    num_plots = len(plot_ready_dict)
    
    # Sets the grid size based on the number of plots
    # Essentially a dynamic system that accomodates all of our subtasks labels (e.g. DBO has 4)
    if num_plots == 1:
        nrows, ncols = 1, 1
        figsize = (10, 6) if figsize == (16, 10) else figsize
    elif num_plots == 2:
        nrows, ncols = 1, 2
    elif num_plots <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = (num_plots + 1) // 2
        ncols = 2
        
    # Creates the figure and axes
    figure, axis_grid = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axis_grid.flatten()
    
    # Color palettes for different plots
    subplot_palettes = ["Blues_r", "Reds_r", "Greens_r", "Oranges_r", "Purples_r", "Greys_r"]
    
    plotted_count = 0
    
    # Loops through the n-gram frequency distributions and plots them
    for i, (plot_label, freq_dist) in enumerate(plot_ready_dict.items()):
        if plotted_count >= nrows * ncols:
            break
            
        axis = flat_axes[plotted_count]
        
        # Creates a DataFrame for the top n-grams
        column_names = ["N-gram", "Frequency"]
        top_items_dataframe = pd.DataFrame(freq_dist.most_common(n_items), columns=column_names)
        
        if not top_items_dataframe.empty:
            # Selecting a color palette
            palette_index = i % len(subplot_palettes)
            current_palette = subplot_palettes[palette_index]
            
            # Creates a bar plot
            sns.barplot(
                x=column_names[1],
                y=column_names[0],
                data=top_items_dataframe,
                ax=axis,
                hue=column_names[0],
                palette=current_palette,
                legend=False
            )
            
            # Adds a grid if enabled
            if show_grid:
                axis.grid(True, linestyle="--", alpha=0.7, axis="x")
                
            # Sets the plot labels and formatting
            axis.set_title(f"Top {n_items} N-grams: \"{plot_label}\"", fontsize=12)
            axis.set_xlabel("Frequency")
            axis.set_ylabel("N-gram" if plotted_count % ncols == 0 else "")
            axis.tick_params(axis="y", labelsize=10)
            axis.tick_params(axis="x", labelsize=10)
            plotted_count += 1
        else:
            # Handles cases with insufficient data and plots a placeholder instead
            axis.set_title(f"\"{plot_label}\" (No significant n-grams)")
            axis.text(0.5, 0.5, "Not enough data\nto plot frequencies", 
                   horizontalalignment="center", verticalalignment="center",
                   transform=axis.transAxes)
            axis.set_xticks([])
            axis.set_yticks([])
            plotted_count += 1
    
    # Removes any unused subplots from the figure
    for j in range(plotted_count, nrows * ncols):
        figure.delaxes(flat_axes[j])
        
    plt.tight_layout()
    
    # Adds a title to the figure if specified
    if title:
        figure.suptitle(title, fontsize=16, y=1.02)
        plt.subplots_adjust(top=0.95)
    
    console.print("N-gram frequency plot created successfully.", style="bold green")
    return figure, axis_grid

# Creates a line plot for n-gram frequency distributions
def create_ngram_frequency_line_plot(ngram_freq_dict, title="N-gram Frequency Line Plot", n_items=15, figsize=(16, 10), show_grid=True):
    # e.g. {C2A_unfiltered: FreqDist, C2A_filtered: FreqDist} will produce 2 plots

    # Validates the input parameters
    if not ngram_freq_dict:
        console.print("Error: No n-gram frequency distributions provided.", style="bold red")
        return None, None
    
    # Converts the n-gram tuples to strings for plotting
    plot_ready_dict = {}
    for label, freq_dist in ngram_freq_dict.items():
        # Creates a new frequency distribution with string keys
        converted_dist = FreqDist()
        for ngram_tuple, freq in freq_dist.items():
            ngram_str = " ".join(ngram_tuple)
            converted_dist[ngram_str] = freq
        
        plot_ready_dict[label] = converted_dist

    # Calculates the grid size based on the number of plots
    num_plots = len(plot_ready_dict)
    
    # Sets the grid size based on the number of plots
    # Essentially a dynamic system that accomodates all of our subtasks labels (e.g. DBO has 4)
    if num_plots == 1:
        nrows, ncols = 1, 1
        figsize = (10, 6) if figsize == (16, 10) else figsize
    elif num_plots == 2:
        nrows, ncols = 1, 2
    elif num_plots <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = (num_plots + 1) // 2
        ncols = 2
        
    # Note: We set squeeze=False to always return a 2D array of axes, even when nrows=ncols=1
    figure, axis_grid = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axis_grid.flatten()
    
    # Color palettes for different distribution plots
    line_colors = ["blue", "red", "green", "orange", "purple", "gray"]
    marker_styles = ["o", "s", "^", "D", "*", "x"] # o=circle, s=square, ^=triangle, D=diamond, *=star, x=x-mark
    
    plotted_count = 0
    
    # Loops through the n-gram frequency distributions and plots them
    for i, (plot_label, freq_dist) in enumerate(plot_ready_dict.items()):
        # Breaks if all subplots have been plotted
        if plotted_count >= nrows * ncols:
            break
            
        # Gets the current axis for the subplot
        axis = flat_axes[plotted_count]
        
        # Creates a DataFrame of the top N n-grams we want to plot
        column_names = ["N-gram", "Frequency"]
        top_items_dataframe = pd.DataFrame(freq_dist.most_common(n_items), columns=column_names)
        
        if not top_items_dataframe.empty:
            # Selects a color and marker style for the current subplot
            color_index = i % len(line_colors)
            marker_index = i % len(marker_styles)
            current_color = line_colors[color_index]
            current_marker = marker_styles[marker_index]
            
            # Creates a line plot with the selected color and marker style
            axis.plot(
                range(len(top_items_dataframe)), 
                top_items_dataframe[column_names[1]],
                marker=current_marker,
                color=current_color,
                linestyle="-",
                linewidth=2,
                markersize=6
            )
            
            # Adds the n-gram labels to the points
            for j, ngram in enumerate(top_items_dataframe[column_names[0]]):
                axis.annotate(
                    ngram, 
                    (j, top_items_dataframe[column_names[1]].iloc[j]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=8,
                    rotation=45
                )
            
            # Adds a grid if enabled
            if show_grid:
                axis.grid(True, linestyle="--", alpha=0.7)
                
            # Sets the plot labels and formatting options
            axis.set_title(f"Top {n_items} N-grams: \"{plot_label}\"", fontsize=12)
            axis.set_xlabel("Rank")
            axis.set_ylabel(column_names[1] if plotted_count % ncols == 0 else "")
            axis.tick_params(axis="y", labelsize=10)
            axis.tick_params(axis="x", labelsize=10)
            axis.set_xticks(range(len(top_items_dataframe)))
            
            # Sets rank labels on the x-axis from 1..N
            x_tick_labels = []
            for idx in range(len(top_items_dataframe)):
                x_tick_labels.append(f"{idx+1}")
            axis.set_xticklabels(x_tick_labels, rotation=45)
            plotted_count += 1
        else:
            # Handles cases with insufficient data and plots a placeholder instead
            axis.set_title(f"\"{plot_label}\" (No significant n-grams)")
            axis.text(0.5, 0.5, "Not enough data\nto plot frequencies",
                   horizontalalignment="center", verticalalignment="center",
                   transform=axis.transAxes)
            axis.set_xticks([])
            axis.set_yticks([])
            plotted_count += 1
    
    # Removes any unused subplots from the figure
    for j in range(plotted_count, nrows * ncols):
        figure.delaxes(flat_axes[j])
   
    plt.tight_layout()
    
    # Adds a title to the figure if specified
    if title:
        figure.suptitle(title, fontsize=16, y=1.02)
        plt.subplots_adjust(top=0.95)
    
    return figure, axis_grid


# Creates a plot for the token frequency distributions
def create_frequency_distribution_plot(freq_dist_dict, title="Frequency Distribution", n_words=15, figsize=(16, 10), column_names=["Word", "Frequency"], show_grid=False):
    # e.g. {C2A_unfiltered: FreqDist, C2A_filtered: FreqDist} will produce 2 plots
    # Calculates the grid size based on the number of plots
    num_plots = len(freq_dist_dict)

    # Dynamic grid size based on the number of plots
    if num_plots == 1:
        nrows, ncols = 1, 1
        figsize = (10, 6) if figsize == (16, 10) else figsize
    elif num_plots == 2:
        nrows, ncols = 1, 2
    elif num_plots <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = (num_plots + 1) // 2
        ncols = 2

    # Note: We set squeeze=False to always return a 2D array of axes, even when nrows=ncols=1
    figure, axis_grid = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axis_grid.flatten()

    # Color palettes for different plots
    subplot_palettes = ["Blues_r", "Reds_r", "Greens_r", "Oranges_r", "Purples_r", "Greys_r"]


    plotted_count = 0

    # Plots each frequency distribution
    for i, (plot_label, freq_dist_object) in enumerate(freq_dist_dict.items()):
        # Break if all subplots have been plotted
        if plotted_count >= nrows * ncols:
            break
        
        # Gets the current axis
        axis = flat_axes[plotted_count]
        
        # Creates a DataFrame of the top N words we want to plot
        top_items_dataframe = pd.DataFrame(freq_dist_object.most_common(n_words), columns=column_names)
        
        if not top_items_dataframe.empty:
            # Selecting a color palette
            palette_index = i % len(subplot_palettes)
            current_palette_name = subplot_palettes[palette_index]

            # Creates a bar plot
            sns.barplot(
                x=column_names[1],
                y=column_names[0],
                data=top_items_dataframe,
                ax=axis,
                hue=column_names[0],  # Uses the Word column as hue for gradient effect
                palette=current_palette_name,
                legend=False
            )
            
            # Adds grid if enabled
            if show_grid:
                axis.grid(True, linestyle="--", alpha=0.7, axis="x")
            
            # Sets the plot labels and formatting options
            axis.set_title(f"Top {n_words} Words: \"{plot_label}\"", fontsize=12)
            axis.set_xlabel(column_names[1])
            axis.set_ylabel(column_names[0] if plotted_count % ncols == 0 else "")
            axis.tick_params(axis="y", labelsize=10)
            axis.tick_params(axis="x", labelsize=10)
            plotted_count += 1
        else:
            # Handles cases with insufficient data and plots a placeholder instead
            axis.set_title(f"\"{plot_label}\" (No significant words)")
            axis.text(0.5, 0.5, "Not enough data\nto plot frequencies", 
                   horizontalalignment="center", verticalalignment="center",
                   transform=axis.transAxes)
            axis.set_xticks([])
            axis.set_yticks([])
            plotted_count += 1
   
    # Removes unused subplots
    for j in range(plotted_count, nrows * ncols):
        figure.delaxes(flat_axes[j])
   
    plt.tight_layout()
    
    # Adds a title to the figure if specified
    if title:
        figure.suptitle(title, fontsize=16, y=1.02)
        plt.subplots_adjust(top=0.95)
    
    return figure, axis_grid


# Creates a line plot showing frequency distributions
def create_frequency_distribution_line_plot(freq_dist_dict, title="Frequency Distribution Line Plot", n_words=15, figsize=(16, 10), column_names=["Word", "Frequency"], show_grid=True):
    # e.g. {C2A_unfiltered: FreqDist, C2A_filtered: FreqDist} will produce 2 plots
    # Calculates the grid size based on the number of plots
    num_plots = len(freq_dist_dict)

    # Sets the grid size based on the number of plots
    if num_plots == 1:
        nrows, ncols = 1, 1
        figsize = (10, 6) if figsize == (16, 10) else figsize
    elif num_plots == 2:
        nrows, ncols = 1, 2
    elif num_plots <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = (num_plots + 1) // 2
        ncols = 2

    # Note: We set squeeze=False to always return a 2D array of axes, even when nrows=ncols=1
    figure, axis_grid = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axis_grid.flatten()

    # Color palettes for different plots
    line_colors = ["blue", "red", "green", "orange", "purple", "gray"]
    marker_styles = ["o", "s", "^", "D", "*", "x"] # o=circle, s=square, ^=triangle, D=diamond, *=star, x=x-mark

    plotted_count = 0

    # Plots each frequency distribution
    for i, (plot_label, freq_dist_object) in enumerate(freq_dist_dict.items()):
        # Break if all subplots have been plotted
        if plotted_count >= nrows * ncols:
            break

        # Gets the current axis
        axis = flat_axes[plotted_count]

        # Creates a DataFrame of the top N words we want to plot
        top_items_dataframe = pd.DataFrame(freq_dist_object.most_common(n_words), columns=column_names)

        if not top_items_dataframe.empty:
            # Selecting a color and marker style for the current subplot
            color_index = i % len(line_colors)
            marker_index = i % len(marker_styles)
            current_color = line_colors[color_index]
            current_marker = marker_styles[marker_index]

            # Creates a line plot
            axis.plot(
                range(len(top_items_dataframe)), 
                top_items_dataframe[column_names[1]], 
                marker=current_marker,
                color=current_color,
                linestyle="-",
                linewidth=2,
                markersize=6
            )

            # Adds word labels to points
            for j, word in enumerate(top_items_dataframe[column_names[0]]):
                axis.annotate(
                    word, 
                    (j, top_items_dataframe[column_names[1]].iloc[j]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=8,
                    rotation=45
                )

            # Adds grid if enabled
            if show_grid:
                axis.grid(True, linestyle="--", alpha=0.7)

            # Sets the plot labels and formatting options
            axis.set_title(f"Top {n_words} Words: \"{plot_label}\"", fontsize=12)
            axis.set_xlabel("Rank")
            axis.set_ylabel(column_names[1] if plotted_count % ncols == 0 else "")
            axis.tick_params(axis="y", labelsize=10)
            axis.tick_params(axis="x", labelsize=10)
            axis.set_xticks(range(len(top_items_dataframe)))
            
            # Sets rank labels on the x-axis from 1..N
            x_tick_labels = []
            for idx in range(len(top_items_dataframe)):
                x_tick_labels.append(f"{idx+1}")
            axis.set_xticklabels(x_tick_labels, rotation=45)
            plotted_count += 1
        else:
            # Handles cases with insufficient data (Fallback)
            axis.set_title(f"\"{plot_label}\" (No significant words)")
            axis.text(0.5, 0.5, "Not enough data\nto plot frequencies",
                   horizontalalignment="center", verticalalignment="center",
                   transform=axis.transAxes)
            axis.set_xticks([])
            axis.set_yticks([])
            plotted_count += 1
   
    # Removes unused subplots
    for j in range(plotted_count, nrows * ncols):
        figure.delaxes(flat_axes[j])
   
    plt.tight_layout()
    
    # Adds a title to the figure if specified
    if title:
        figure.suptitle(title, fontsize=16, y=1.02)
        plt.subplots_adjust(top=0.95)
    
    return figure, axis_grid
