# GLMM Target Encoders for High-Dimensional Categorical Features

This repo carries a Tensorflow implementation of the Generalized Linear Mixed Effect Model (GLMM) encoders described in
[Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features](https://arxiv.org/pdf/2104.00629.pdf)
by Pargent et al.

The GLMM models are described using Tensorflow Probability (TFP) distributions and are fit
using variational inference with a mean-field variational posterior over model parameters.


## Usage

The `glmm_encoder/encoders` package carries TF based GLMM target encoders for regression, binary classification and 
multiclass classification tasks. These encoders conform to the `tf.keras.Model` API and can be compiled, fit and 
used for inference like any other TF model.

#### Example:

```python
from glmm_encoder.examples.dataset_utils import load_toy_regression_dataset
from glmm_encoder.examples.model_utils import log_likelihood_loss_plot
from glmm_encoder.encoders import GLMMRegressionFeatureEncoder
import numpy as np
import tensorflow as tf
import pandas as pd

if __name__ == "__main__":
    dataset = load_toy_regression_dataset(seed=22)
    targets = dataset[["y"]].astype(np.float32).values.flatten()
    features = dataset[["x"]].astype(int).x.values
    n_levels = int(dataset[["x"]].nunique())

    model = GLMMRegressionFeatureEncoder(n_levels)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-2))
    history = model.fit(features, targets, batch_size=1000, epochs=100)
    pred_inputs = list(range(0, n_levels + 10))
    predictions = pd.DataFrame(
        list(zip(pred_inputs, model.predict(pred_inputs).flatten().tolist())),
        columns=["Feature Level", "Encoded value"]
    )
    print(f"\nPredictions:\n{predictions.to_string()}\n")
    model.print_posterior_estimates()
    log_likelihood_loss_plot(history.history["loss"])
```

For similar examples for binary and multiclass classification tasks, take a look at 
`glmm_encoder/examples/binary_classification/toy_bin_classification_example.py`
and `glmm_encoder/examples/multiclass_classification/toy_multi_classification_example.py`


#### Tests, linting, type checking and docstrings
Unit tests... don't exist at the moment and need to be added.
Type checking can be run using `make test-mypy`.
Docstyle checking can be run using `make test-docstyle`.
Linting can be run using `make test-pylint`.
To run all of these steps run `make test`.


## Benchmarks
### Comparing GLMM target encoders to others
Pargent et al. compare several categorical feature encoding types across datasets with using their R implementation. 
We do something similar with the TF implementation, but using only a single dataset per task (regression, 
binary classification, multiclass classification). The results of our comparison are shown below:

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
`glmm_encoder/examples/regression/toy_bin_classification_example.py` and
`glmm_encoder/examples/regression/toy_regression_example.py`. Specifically, the `--compare_with_R` flag, when supplied
to these scripts, runs a single run in both TF and R (via rpy2), with a fixed dataset. In order to change the dataset
used, change the seed supplied to the dataset generating functions.

## Caveats
### Distributed training
Since all GLMM encoders implemented here derive from `tf.model.Keras` and are gradient based,
they _should_ work with any distribution strategy used to train TF models. That being said, these encoders have not been
tested with distributed training, and it is not clear how the VI inference methods interact with distributed settings.
This requires more exploration.

### TF Warnings about auto-vectorized joint distributions
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
