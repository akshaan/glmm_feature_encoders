# GLMM Target Encoders for High-Dimensional Categorical Features

This repo carries a Tensorflow implementation of the GLMM encoders described in
[Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features](https://arxiv.org/pdf/2104.00629.pdf)
by Pargent et al.

The GLMM models are described using Tensorflow Probability (TFP) distributions and are fit
using variational inference with a mean-field variational posterior over model paramters.

### TODOs:
- [] Test with real datasets (avocado-sales, Midwest_survey)
- [] Add cross validation support?
- [] Try distributed training and add example
- [] Typing, linting, tests, docstrings
- [] Timing and accuracy comparison with lme4
