from typing import List, Optional

import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from glmm_encoder.examples.dataset_utils import Dataset
from collections import defaultdict
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt


class Model(ABC):
    @abstractmethod
    def train(self, train_x: np.array, train_y: np.array) -> None:
        pass

    @abstractmethod
    def predict(self, test_x: np.array) -> np.array:
        pass


def one_hot_encoder_model_preds(dataset: Dataset, model: Model) -> List[float]:
    col_to_encode = dataset.train_features[[dataset.col_to_encode]]
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore").fit(col_to_encode)
    feature_names = [x for x in dataset.feature_names if x != dataset.col_to_encode]
    train_features_np = np.concatenate([
        dataset.train_features[feature_names].to_numpy(),
        encoder.transform(col_to_encode)
    ], axis=1)
    model.train(train_features_np, dataset.train_labels)
    test_features_np = np.concatenate([
        dataset.test_features[feature_names].to_numpy(),
        encoder.transform(dataset.test_features[[dataset.col_to_encode]])
    ], axis=1)
    return model.predict(test_features_np)


def ordinal_encoder_model_preds(dataset: Dataset, model: Model) -> List[float]:
    col_to_encode = dataset.train_features[[dataset.col_to_encode]]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(col_to_encode)
    feature_names = [x for x in dataset.feature_names if x != dataset.col_to_encode]
    train_features_np = np.concatenate([
        dataset.train_features[feature_names].to_numpy(),
        encoder.transform(col_to_encode)
    ], axis=1)
    model.train(train_features_np, dataset.train_labels)
    test_features_np = np.concatenate([
        dataset.test_features[feature_names].to_numpy(),
        encoder.transform(dataset.test_features[[dataset.col_to_encode]])
    ], axis=1)
    return model.predict(test_features_np)


def impact_encoder_model_preds(dataset: Dataset, model: Model, multiclass: bool = False,
                               num_classes: Optional[int] = None) -> List[float]:
    col_to_encode = dataset.train_features[[dataset.col_to_encode]].values
    train_labels = dataset.train_labels.values
    if not multiclass:
        level_to_labels_dict = defaultdict(list)
        for level, label in zip(col_to_encode, train_labels):
            level_to_labels_dict[level[0]].append(label)
        encodings_dict = defaultdict(lambda: [0.])
        for level, labels in level_to_labels_dict.items():
            encodings_dict[level] = [np.mean(labels)] if labels else [0.]
        feature_names = [x for x in dataset.feature_names if x != dataset.col_to_encode]
        train_features_np = np.concatenate([
            dataset.train_features[feature_names],
            [encodings_dict[x[0]] for x in col_to_encode]
        ], axis=1)
        model.train(train_features_np, dataset.train_labels)
        test_features_np = np.concatenate([
            dataset.test_features[feature_names].to_numpy(),
            [encodings_dict[x[0]] for x in dataset.test_features[[dataset.col_to_encode]].values]
        ], axis=1)
        return model.predict(test_features_np)
    else:
        if not num_classes:
            raise ValueError("num_classes must be specified for multiclass mode")
        level_to_labels_dict = defaultdict(lambda: defaultdict(list))
        for level, label in zip(col_to_encode, train_labels):
            level_to_labels_dict[level[0]][label].append(1.)
        encodings_dict = defaultdict(lambda: defaultdict(float))
        for level, labels in level_to_labels_dict.items():
            for cls in range(num_classes):
                encodings_dict[level][cls] = np.mean(labels[cls]) if labels[cls] else 0.
        feature_names = [x for x in dataset.feature_names if x != dataset.col_to_encode]
        train_features_np = np.concatenate([
            dataset.train_features[feature_names],
            [[encodings_dict[x[0]][cls] for cls in range(num_classes)] for x in col_to_encode]
        ], axis=1)
        model.train(train_features_np, dataset.train_labels)
        test_features_np = np.concatenate([
            dataset.test_features[feature_names].to_numpy(),
            [[encodings_dict[x[0]][cls] for cls in range(num_classes)]
             for x in dataset.test_features[[dataset.col_to_encode]].values]
        ], axis=1)
        return model.predict(test_features_np)


def frequency_encoder_model_preds(dataset: Dataset, model: Model) -> List[float]:
    col_to_encode = dataset.train_features[[dataset.col_to_encode]].values
    encodings_dict = defaultdict(float)
    total_count = 0.
    for level in col_to_encode:
        encodings_dict[level[0]] += 1.
        total_count += 1.
    encodings_dict = defaultdict(lambda: [0.], {x: [y / total_count] for x, y in encodings_dict.items()})
    feature_names = [x for x in dataset.feature_names if x != dataset.col_to_encode]
    train_features_np = np.concatenate([
        dataset.train_features[feature_names],
        [encodings_dict[x[0]] for x in col_to_encode]
    ], axis=1)
    model.train(train_features_np, dataset.train_labels)
    test_features_np = np.concatenate([
        dataset.test_features[feature_names].to_numpy(),
        [encodings_dict[x[0]] for x in dataset.test_features[[dataset.col_to_encode]].values]
    ], axis=1)
    return model.predict(test_features_np)


def log_likelihood_loss_plot(losses: List[float]):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(losses, 'k-')
    ax.set(xlabel="Iteration",
           ylabel="Loss (ELBO)",
           title="Loss during training",
           ylim=0)
    plt.show()
