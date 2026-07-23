# Baseline system - Attacks on the Basic Democratic Order

A simple classification approach based on a **linear Support Vector Machine (SVM)** (Chang & Lin, 2011) with TF-IDF weighted bag-of-phrases (unigrams and bigrams) was chosen as the baseline system. The vocabulary was limited to the 5,000 most frequent terms in the training dataset. To counteract class imbalance, a cost-sensitive SVM (weighted SVM) was used, in which misclassification of instances from underrepresented classes is more penalised. 

The Jupyter notebook `dbo_baseline.ipynb` contains the code for training the model and for making predictions on unseen data.

## Results 

The system achieved the following metrics on the test data, with the Macro-F1 metric being decisive for the ranking on the leaderboard:

| Category      |   P  |   R  |  F1  |
| ------------- | ---- | ---- |  -:  |
| agitation     | 0.29 | 0.46 | 0.36 |
| criticism     | 0.36 | 0.65 | 0.47 |
| nothing       | 0.94 | 0.82 | 0.88 |
| subversive    | 0.60 | 0.12 | 0.20 |
| **Mac. avg.** | **0.55** | **0.51** | **0.47** |
| **Weight. avg.** | 0.85 | 0.78 | 0.80 |

## References

- Chang, C.-C., & Lin, C.-J. (2011). LIBSVM: A library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2(3). https://doi.org/10.1145/1961189.1961199
