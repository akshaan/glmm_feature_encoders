# GLMM Target Encoders for High-Dimensional Categorical Features

This repo carries a Tensorflow implementation of the Generalized Linear Mixed Effect Model (GLMM) encoders for high-dimensional categorical features described in
[Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features](https://arxiv.org/pdf/2104.00629.pdf)
by Pargent et al.

The GLMM models are described using Tensorflow Probability (TFP) distributions and are fit
using variational inference with a mean-field variational posterior over model parameters.


## Usage

## Benchmarks
### Comparing GLMM target encoders to others
Pargent et al. compare several categorical feature encoding types across datasets with using their R implementation. We do something similar with the TF implementation,
but using only a single dataset per task (regression, binary classification, multiclass classification). The results of our comparison are shown below:

![Alt text](./Figure_1.png?raw=true "Figure 1.")

To re-compute these benchmarks with different parameters etc. take a look at `glmm_encoder/examples/compare_encoders.py`

### Comparing this implementation to R (lme4)
Pargent et al.'s implementation uses R with the lme4 package for fitting generalized linear mixed effect models. We
compare the fit of our TF based GLMMs with lme4 for the binary classification and regression tasks. We use data drawn
from a known distribution in both cases. The results of our comparison are below:

#### Regression:
|Parameter          |True Value|R (lme4) estimate|Tensorflow VI estimate|
|-------------------|----------|-----------------|----------------------|
|Global Intercept   |1.0       |1.0334           |1.0284                |
|Random effect scale|0.5       |0.5041           |0.4946                |

#### Binary Classification:
|Parameter          |True value|R (lme4) estimate|Tensorflow VI estimate|
|-------------------|----------|-----------------|----------------------|
|Global Intercept   |1.0       |1.053            |1.0626                |
|Random effect scale|0.5       |0.5041           |0.4937                |


N.b. these comparisons were made over a small number of runs. To re-compute this benchmark, take a look at the
`glmm_encoders/examples/regression/toy_bin_classification_example.py` and
`glmm_encoders/examples/regression/toy_regression_example.py`. Specifically, the `--compare_with_R` flag, when supplied
to these scripts, runs a single run in both TF and R (via rpy2), with a fixed dataset. In order to change the dataset
used, change the seed supplied to the dataset generating functions.

## Caveats
### Distributed training
Since all GLMM encoders implemented here derive from `tf.model.Keras` and are gradient based,
they _should_ work with any distribution strategy used to train TF models. That being said, these encoders have not been
tested with distributed training, and it is not clear how the VI inference methods interact with distributed settings.
This requires more exploration.

### TF Warnings about auto-vectorized join distributions
Currently, the GLMM encoders emit warnings of the form:
```buildoutcfg
UserWarning: Saw Tensor seed 
    Tensor(
        "monte_carlo_variational_loss/expectation/JointDistributionSequentialAutoBatched/log_prob/Const:0", 
         shape=(2,), dtype=int32
    ),
    implying stateless sampling. 
Autovectorized functions that use stateless sampling may be quite slow because the current implementation falls back
to an explicit loop. This will be fixed in the future. For now, you will likely see better performance from stateful
sampling, which you can invoke by passing a Python `int` seed.
```

These warnings should be fixed in a future version as noted in the message. In the meantime, it might be possible to
pass a seed into the sampling invocations, but this requires some investigation.

### Multiclass classification is slow
Multiclass classification trains several binary GLMM encoders (one per class). This is pretty slow, especially for many
output classes or large datasets. It might be possible to speed this up in the future.

### Tests, type checking and docstrings
These... don't exist at the moment and need to be added.
