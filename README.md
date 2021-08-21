# GLMM Target Encoders for High-Dimensional Categorical Features

This repo carries a Tensorflow implementation of the GLMM encoders described in
[Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features](https://arxiv.org/pdf/2104.00629.pdf)
by Pargent et al.

The GLMM models are described using Tensorflow Probability (TFP) distributions and are fit
using variational inference with a mean-field variational posterior over model paramters.

### TODOs:
- [] Clean up encoder APIs (and maybe conform to tf.keras.Model?)
- [] Add support for out-of-sample levels for features
- [] Add support for multiclass classification
- [] Add examples
- [] Test with real datasets
- [] Add cross validation support
- [] Fix classfication mode
- [] Distributed training
- [] Utils for model criticism
- [] Typing, linting, tests, docstrings