"""Util functions for dataset loading"""

import numpy as np
import pandas as pd
from itertools import chain


def load_toy_regression_dataset(feature_levels=25, samples_per_level=1000, seed=None):
    if seed:
        np.random.seed(seed)
    fixed_effect = 1
    noise_scale = 0.1
    y = []
    random_intercepts_for_level = np.random.normal(0, 0.5, feature_levels)
    for level in range(feature_levels):
        for _ in range(samples_per_level):
            random_effect = random_intercepts_for_level[level]
            samples_for_level = random_effect + fixed_effect + np.random.normal(0, noise_scale, 1)
            y += samples_for_level.tolist()
    x = chain(*[[level] * samples_per_level for level in range(feature_levels)])

    return pd.DataFrame({"x": x, "y": y})


def load_toy_binary_classification_dataset(feature_levels=25, samples_per_level=1000, seed=None):
    if seed:
        np.random.seed(seed)
    fixed_effect = 1
    noise_scale = 0.1
    y = []
    random_intercepts_for_level = np.random.normal(0, 0.5, feature_levels)
    for level in range(feature_levels):
        for _ in range(samples_per_level):
            random_effect = random_intercepts_for_level[level]
            logit = random_effect + fixed_effect + np.random.normal(0, noise_scale, 1)
            p = np.exp(-np.logaddexp(0, -logit))  # Sigmoid
            samples_for_level = np.random.binomial(1, p)
            y += samples_for_level.tolist()
    x = chain(*[[level] * samples_per_level for level in range(feature_levels)])

    return pd.DataFrame({"x": x, "y": y})

def load_toy_multiclass_classification_dataset(feature_levels=25, samples_per_level=1000, seed=None):
    pass