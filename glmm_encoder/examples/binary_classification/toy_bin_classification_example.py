from glmm_encoder.examples.dataset_utils import load_toy_binary_classification_dataset
from glmm_encoder.examples.model_utils import log_likelihood_loss_plot
from glmm_encoder.encoders import GLMMBinaryTargetEncoder
import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import tempfile
from pathlib import Path
import shutil
from rpy2 import robjects

parser = argparse.ArgumentParser(description="Toy Binary Classification Examples for GLMM Feature Encoders")
parser.add_argument("--compare_to_R", help="Compare to R implementation using lme4", action="store_true")

tfd = tfp.distributions
tfb = tfp.bijectors

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = load_toy_binary_classification_dataset(seed=22)
    targets = dataset[["y"]].astype(np.float32).values.flatten()
    features = dataset[["x"]].astype(int).x.values
    n_levels = int(dataset[["x"]].nunique())

    model = GLMMBinaryTargetEncoder(n_levels)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-2))
    history = model.fit(features, targets, batch_size=2500, epochs=50)
    pred_inputs = list(range(0, n_levels + 10))
    predictions = pd.DataFrame(
        list(zip(pred_inputs, model.predict(pred_inputs).flatten().tolist())),
        columns=["Feature Level", "Encoded value"]
    )
    print(f"\nPredictions:\n{predictions.to_string()}\n")
    model.print_posterior_estimates()
    log_likelihood_loss_plot(history.history["loss"])

    if args.compare_to_R:
        # Write dataset
        tempdir = tempfile.mkdtemp()
        dataset.to_csv(tempdir / "bin_classification_dataset.csv")

        # Run R code
        r_script_path = Path(__file__).parent / "r_scripts/binary_classif_lmer.R"
        with open(r_script_path, "r") as script_file:
            robjects.r(script_file.read())

        print("\n Model fit using R (lme4):")
        print(robjects.r["fit"].r_repr())

        # Clean up
        shutil.rmtree(tempdir)


