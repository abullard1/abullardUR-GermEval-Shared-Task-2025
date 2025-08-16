# Baseline system - Violence 

The **Large Language Model Qwen2.5**, with 32 billion parameters, was used as the baseline for violence detection (Yang et al., 2024). The LLM was tasked with binary classification of tweets in a few-shot scenario. The LLM was assigned the role of a filter system specialised in detecting harmful content on social media via a system prompt. It was also provided with a definition of ‘call to definition’ that was essentially the same as the one given to the annotators. The examples required for in-context learning are the same as those available on the website and presented to the annotators. The script is intended for LLMs running locally with Ollama. 

The R notebook `vio_baseline.html` contains the code for classification with the LLM and the output as a formatted HTML document. The source code can be found in the RMD file `vio_baseline.RMD`. 

## Results 

The system achieved the following metrics on the test data, with the Macro-F1 metric being decisive for the ranking on the leaderboard. 


| Category      |   P  |   R  |  F1  |
| ------------- | ---- | ---- |  -:  |
| true          | 0.31 | 0.93 | 0.47 |
| false         | 0.99 | 0.84 | 0.91 |
| **Mac. avg.** | **0.65** | **0.88** | **0.69** |
| **Weight. avg.** | 0.94 | 0.85 | 0.88 |

## References

-  Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., … Qiu, Z. (2024, September). Qwen2.5 Technical Report. https://qwenlm.github.io/blog/qwen2.5/
