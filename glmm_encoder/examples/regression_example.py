from glmm_encoder.datasets.datasets import (
    load_toy_regression_dataset
)
from glmm_encoder.viz.viz import log_likelihood_loss_plot
from glmm_encoder.encoders import GLMMRegressionTargetEncoder
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors

if __name__ == "__main__":
    dataset = load_toy_regression_dataset(seed=22)
    targets = dataset[["y"]].astype(np.float32).values.flatten()
    features = dataset[["x"]].astype(int)
    n_levels = int(features.nunique())

    model = GLMMRegressionTargetEncoder(n_levels)
    losses = model.fit(features.x.values, targets)
    model.predict(features)
    log_likelihood_loss_plot(losses)

