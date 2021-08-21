from typing import List, Dict, Callable

import math
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error

from glmm_encoder.datasets.datasets import Dataset
from glmm_encoder.encoders.encoders import GLMMRegressionTargetEncoder
from collections import defaultdict
from xgboost import XGBModel


def one_hot_encoder_model_preds(dataset: Dataset,
                                train_model_fn: Callable[[np.array, np.array], XGBModel]) -> List[float]:
    col_to_encode = dataset.train_features[[dataset.col_to_encode]]
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore").fit(col_to_encode)
    feature_names = [x for x in dataset.feature_names if x != dataset.col_to_encode]
    train_features_np = np.concatenate([
        dataset.train_features[feature_names].to_numpy(),
        encoder.transform(col_to_encode)
    ], axis=1)
    model = train_model_fn(train_features_np, dataset.train_labels)
    test_features_np = np.concatenate([
        dataset.test_features[feature_names].to_numpy(),
        encoder.transform(dataset.test_features[[dataset.col_to_encode]])
    ], axis=1)
    return model.predict(test_features_np)


def ordinal_encoder_model_preds(dataset: Dataset,
                                train_model_fn: Callable[[np.array, np.array], XGBModel]) -> List[float]:
    col_to_encode = dataset.train_features[[dataset.col_to_encode]]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(col_to_encode)
    feature_names = [x for x in dataset.feature_names if x != dataset.col_to_encode]
    train_features_np = np.concatenate([
        dataset.train_features[feature_names].to_numpy(),
        encoder.transform(col_to_encode)
    ], axis=1)
    model = train_model_fn(train_features_np, dataset.train_labels)
    test_features_np = np.concatenate([
        dataset.test_features[feature_names].to_numpy(),
        encoder.transform(dataset.test_features[[dataset.col_to_encode]])
    ], axis=1)
    return model.predict(test_features_np)


def impact_encoder_model_preds(dataset: Dataset,
                               train_model_fn: Callable[[np.array, np.array], XGBModel]) -> List[float]:
    col_to_encode = dataset.train_features[[dataset.col_to_encode]].values
    train_labels = dataset.train_labels.values
    level_to_labels_dict = defaultdict(list)
    for level, label in zip(col_to_encode, train_labels):
        level_to_labels_dict[level[0]].append(label)
    encodings_dict = defaultdict(lambda: [0])
    for level, labels in level_to_labels_dict.items():
        encodings_dict[level] = [np.mean(labels)]
    feature_names = [x for x in dataset.feature_names if x != dataset.col_to_encode]
    train_features_np = np.concatenate([
        dataset.train_features[feature_names],
        [encodings_dict[x[0]] for x in col_to_encode]
    ], axis=1)
    model = train_model_fn(train_features_np, dataset.train_labels)
    test_features_np = np.concatenate([
        dataset.test_features[feature_names].to_numpy(),
        [encodings_dict[x[0]] for x in dataset.test_features[[dataset.col_to_encode]].values]
    ], axis=1)
    return model.predict(test_features_np)


def frequency_encoder_model_preds(dataset: Dataset,
                                  train_model_fn: Callable[[np.array, np.array], XGBModel]) -> List[float]:
    col_to_encode = dataset.train_features[[dataset.col_to_encode]].values
    encodings_dict = defaultdict(int)
    total_count = 0
    for level in col_to_encode:
        encodings_dict[level[0]] += 1
        total_count += 1
    encodings_dict = defaultdict(lambda: [0], {x: [y/total_count] for x, y in encodings_dict.items()})
    feature_names = [x for x in dataset.feature_names if x != dataset.col_to_encode]
    train_features_np = np.concatenate([
        dataset.train_features[feature_names],
        [encodings_dict[x[0]] for x in col_to_encode]
    ], axis=1)
    model = train_model_fn(train_features_np, dataset.train_labels)
    test_features_np = np.concatenate([
        dataset.test_features[feature_names].to_numpy(),
        [encodings_dict[x[0]] for x in dataset.test_features[[dataset.col_to_encode]].values]
    ], axis=1)
    return model.predict(test_features_np)


def glmm_encoder_model_preds(dataset: Dataset,
                             train_model_fn: Callable[[np.array, np.array], XGBModel]) -> List[float]:
    num_levels = int(dataset.train_features[[dataset.col_to_encode]].nunique())
    col_to_encode = dataset.train_features[[dataset.col_to_encode]].values.flatten()
    encoder = GLMMRegressionTargetEncoder(num_levels)
    encoder.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-2))
    encoder.fit(
        col_to_encode,
        dataset.train_labels.astype(np.float32).values.flatten(),
        batch_size=2000, epochs=500, verbose=False)
    feature_names = [x for x in dataset.feature_names if x != dataset.col_to_encode]
    train_features_np = np.concatenate([
        dataset.train_features[feature_names],
        encoder.predict(col_to_encode)
    ], axis=1)
    model = train_model_fn(train_features_np, dataset.train_labels)
    test_features_np = np.concatenate([
        dataset.test_features[feature_names].to_numpy(),
        encoder.predict(dataset.test_features[[dataset.col_to_encode]].values)
    ], axis=1)
    return model.predict(test_features_np)


def print_scores(dataset: Dataset, preds_dict: Dict[str, List[float]]) -> None:
    encoding_types = []
    mse_values = []
    print(f"Dataset: {dataset.name} (OpenML ID = {dataset.openml_id})")
    for encoding_type, preds in preds_dict.items():
        encoding_types.append(encoding_type)
        mse_values.append(math.sqrt((mean_squared_error(dataset.test_labels, preds))))
    print(pd.DataFrame({"Encoding type": encoding_types, "Root Mean Squared Error": mse_values}))
    print()


