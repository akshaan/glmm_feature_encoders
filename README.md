# GLMM Target Encoders for High-Dimensional Categorical Features

This repo carries a Tensorflow implementation of the GLMM encoders described in
[Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features](https://arxiv.org/pdf/2104.00629.pdf)
by Pargent et al.

The GLMM models are described using Tensorflow Probability (TFP) distributions and are fit
using variational inference with a mean-field variational posterior over model paramters.

### TODOs:
- [] Add support for out-of-sample levels for features
- [] Add support for multiclass classification
- [] Test with real datasets
- [] Add cross validation support
- [] Fix classfication mode
- [] Distributed training
- [] Typing, linting, tests, docstrings
