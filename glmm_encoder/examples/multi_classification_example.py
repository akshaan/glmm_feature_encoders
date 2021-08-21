from glmm_encoder.datasets import load_toy_multiclass_classification_dataset
from glmm_encoder.viz import log_likelihood_loss_plot
from glmm_encoder.encoders import GLMMMulticlassTargetEncoder
import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf
import pandas as pd

tfd = tfp.distributions
tfb = tfp.bijectors

if __name__ == "__main__":
    dataset = load_toy_multiclass_classification_dataset(seed=22)
    targets = dataset[["y"]].astype(np.float32).values.flatten()
    features = dataset[["x"]].astype(int).x.values
    n_levels = int(dataset[["x"]].nunique())
    n_classes = int(dataset[["y"]].nunique())

    model = GLMMMulticlassTargetEncoder(n_levels, n_classes)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-2))
    history = model.fit(features, targets, batch_size=10000, epochs=100)
    pred_inputs = list(range(0, n_levels + 10))
    pred_outputs = model.predict(pred_inputs)
    print(pred_outputs)
    predictions = pd.DataFrame(
        list(zip(pred_inputs, pred_outputs)),
        columns=["Feature Level", "Encoded value"]
    )
    print(f"\nPredictions:\n{predictions.to_string()}\n")
    model.print_posterior_estimates()
    log_likelihood_loss_plot(history.history["loss"])

