"""Util functions for dataset loading"""
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
from itertools import chain
from dataclasses import dataclass
import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


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
    train_labels: pd.Series
    test_features: pd.DataFrame
    test_labels: pd.Series
    col_to_encode: str
    feature_names: List[str]
    name: str
    openml_id: int


def load_openml_dataset(openml_id: Union[int, str],  col_to_encode: str, use_label_encoding: bool = False) -> Dataset:
        response = openml.datasets.get_dataset(openml_id)
        x, y, categorical_indicator, attribute_names = response.get_data(
            target=response.default_target_attribute, dataset_format="dataframe"
        )

        # Clean NaN rows
        rows_with_nan = [index for index, row in x.iterrows() if row.isnull().any()]
        x = x[~x.index.isin(rows_with_nan)]
        y = y[~y.index.isin(rows_with_nan)]

        # Split into train/test sets
        train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size=0.2)

        # Ordinal encode all feature including the one we want to encode with custom encoders
        # This makes it easy to use these columns directly in models. For the custom encoding column,
        # this does not change anything, since a string or other level type for that feature is equivalent
        # to an integer type.
        encoded_train_features = {}
        encoded_test_features = {}
        for feature_name, is_categorical in zip(attribute_names, categorical_indicator):
            if not is_categorical:
                continue
            encoder = OrdinalEncoder(
                dtype=np.int64,
                handle_unknown="use_encoded_value",
                unknown_value=-1).fit(train_features[[feature_name]])
            encoded_train_features[feature_name] = encoder.transform(train_features[[feature_name]])
            encoded_test_features[feature_name] = encoder.transform(test_features[[feature_name]])

        train_features = train_features.assign(**encoded_train_features)
        test_features = test_features.assign(**encoded_test_features)

        if use_label_encoding:
            # Encode string labels as integers
            label_encoder = LabelEncoder().fit(train_labels)
            train_labels = pd.Series(label_encoder.transform(train_labels))
            test_labels = pd.Series(label_encoder.transform(test_labels))

        return Dataset(
            train_features=train_features,
            test_features=test_features,
            train_labels=train_labels,
            test_labels=test_labels,
            col_to_encode=col_to_encode,
            feature_names=attribute_names, name=response.name, openml_id=openml_id)
