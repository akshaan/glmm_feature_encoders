from glmm_encoder.datasets.datasets import (
    load_toy_regression_dataset
)
from glmm_encoder.viz.viz import log_likelihood_loss_plot
from glmm_encoder.encoders import GLMMRegressionTargetEncoder
import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf

tfd = tfp.distributions
tfb = tfp.bijectors

if __name__ == "__main__":
    dataset = load_toy_regression_dataset(seed=22)
    targets = dataset[["y"]].astype(np.float32).values.flatten()
    features = dataset[["x"]].astype(int).x.values
    n_levels = int(dataset[["x"]].nunique())

    model = GLMMRegressionTargetEncoder(n_levels)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-2))
    history = model.fit(features, targets, batch_size=1000, epochs=500)
    model.predict(features)
    model.print_posterior_estimates()
    log_likelihood_loss_plot(history.history["loss"])
