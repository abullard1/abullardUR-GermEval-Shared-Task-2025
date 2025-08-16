# Baseline system - Call to Action

A **gradient-boosting classifier** was trained as the baseline system, utilising sentence embeddings extracted from SentenceBert (Reimers & Gurevych, 2019) and the tweet's polarity as features. Polarity was determined using a lexicon-based approach with the TextBlob programme (Loria, 2025). Undersampling was applied to address the high imbalance of the training dataset. 

The Jupyter notebook `c2a_baseline.ipynb` contains the code for training the model and for making predictions on unseen data.

## Results 

The system achieved the following metrics on the test data, with the Macro-F1 metric being decisive for the ranking on the leaderboard. 

| Category      |   P  |   R  |  F1  |
| ------------- | ---- | ---- |  -:  |
| true          | 0.23 | 0.78 | 0.36 |
| false         | 0.97 | 0.72 | 0.83 |
| **Mac. avg.** | **0.60** | **0.75** | **0.59** |
| **Weight. avg.** | 0.90 | 0.73 | 0.78 |

## References

- Loria, S. (2025). Sloria/TextBlob (Version 0.7.0) [Python]. https://github.com/sloria/TextBlob (Original work published 2013)
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 3982–3992. https://doi.org/10.18653/v1/D19-1410

