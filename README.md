# WIP Text classification using multilingual BERT (mBert)

This repo attempts to reproduce the results presented in [Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT](https://www.aclweb.org/anthology/D19-1077.pdf), 
regarding a zero-shot text classification on MLdoc dataset.

More specifically, the scores for zero-shot cross-lingual transfer included in the original work are the following:
| Language  | Score |
| ------------- | ------------- |
| en  | 94.2  |
| de  | 80.2  |
| es  | 72.6  |
| fr  | 72.6  |
| it  | 68.9  |
| ja  | 56.6 |
| ru  | 73.7  |
| Average  | 74.5  |

In contrast, the scores that we managed to reproduce are the following:

| Language  | Score |
| ------------- | ------------- |
| en  | 96.5  |
| de  | 79.1  |
| es  | 73.4  |
| fr  | 78.0  |
| it  | 65.7  |
| ja  | 71.4 |
| ru  | 62.8  |
| Average  | 75.2  |




