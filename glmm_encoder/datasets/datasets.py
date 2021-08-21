"""Util functions for dataset loading"""
from typing import List

import numpy as np
import pandas as pd
from itertools import chain
from dataclasses import dataclass
import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def load_toy_regression_dataset(
        feature_levels: int = 25,
        samples_per_level: int = 1000,
        seed: float = None) -> pd.DataFrame:
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


def load_toy_binary_classification_dataset(
        feature_levels: int = 25,
        samples_per_level: int = 1000,
        seed: float = None) -> pd.DataFrame:
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


def load_toy_multiclass_classification_dataset(
        feature_levels: int = 25,
        num_classes: int = 5,
        samples_per_level: int = 1000,
        seed: float = None):
    if seed:
        np.random.seed(seed)
    x = chain(*[[level] * samples_per_level for level in range(feature_levels)])
    y = [np.random.randint(0, num_classes) for _ in range(feature_levels * samples_per_level)]

    return pd.DataFrame({"x": x, "y": y})

@dataclass
class Dataset:
    train_features: pd.DataFrame
    train_labels: pd.DataFrame
    test_features: pd.DataFrame
    test_labels: pd.DataFrame
    col_to_encode: str
    feature_names: List[str]
    name: str
    openml_id: int


def __load_openml_dataset(openml_id: int, col_to_encode) -> Dataset:
    response = openml.datasets.get_dataset(openml_id)
    print(f"Loading {response.name} (OpenML Id = {openml_id})")
    X, y, categorical_indicator, attribute_names = response.get_data(
        target=response.default_target_attribute, dataset_format="dataframe"
    )
    print(X)
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2)
    # Ordinal encoder all features barring the one we want to experiment with (i.e. col_to_encode)
    for feature_name, is_categorical in zip(attribute_names, categorical_indicator):
        if not is_categorical:
            continue
        encoder = OrdinalEncoder(dtype=np.int64).fit(train_features[[feature_name]])
        train_features.loc[:, feature_name] = encoder.transform(train_features[[feature_name]])
        test_features.loc[:, feature_name] = encoder.transform(test_features[[feature_name]])

    return Dataset(
        train_features=train_features,
        test_features=test_features,
        train_labels=train_labels,
        test_labels=test_labels,
        col_to_encode=col_to_encode,
        feature_names=attribute_names, name=response.name, openml_id=openml_id)


def load_avocado_sales_dataset() -> Dataset:
    return __load_openml_dataset(41210, col_to_encode="region")


def load_road_safety_dataset() -> Dataset:
    return __load_openml_dataset(42803, col_to_encode="LSOA_of_Accident_Location")
