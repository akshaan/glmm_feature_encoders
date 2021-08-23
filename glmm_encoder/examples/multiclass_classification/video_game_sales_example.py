from typing import List

from glmm_encoder.examples.dataset_utils import Dataset, load_openml_dataset
from glmm_encoder.encoders import GLMMMultiClassifierFeatureEncoder
from xgboost import XGBClassifier
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import pandas as pd
import argparse


from glmm_encoder.examples.model_utils import (
    frequency_encoder_model_preds,
    impact_encoder_model_preds,
    ordinal_encoder_model_preds,
    one_hot_encoder_model_preds,
    Model
)

parser = argparse.ArgumentParser(description='Video Genre Multiclass Classification Task')
parser.add_argument("--nruns", type=int, help="number of runs", default=1)
parser.add_argument("--outpath", type=str, help="output path for scores from model runs", default=None)


class MultiClassifier(Model):
    def __init__(self):
        self.xgb_model = XGBClassifier(n_estimators=10, verbosity=0, use_label_encoder=False)

    def train(self, train_x: np.array, train_y: np.array) -> None:
        self.xgb_model.fit(train_x, train_y)

    def predict(self, test_x: np.array) -> np.array:
        return self.xgb_model.predict_proba(test_x)


def glmm_encoder_model_preds(dataset: Dataset, model: Model) -> List[float]:
    col_to_encode = dataset.train_features[[dataset.col_to_encode]].values.flatten()
    n_levels = int(dataset.train_features[[dataset.col_to_encode]].nunique())
    n_classes = int(dataset.train_labels.nunique())
    encoder = GLMMMultiClassifierFeatureEncoder(num_levels=n_levels, num_classes=n_classes)
    encoder.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-2))
    encoder.fit(
        col_to_encode,
        dataset.train_labels.astype(np.float32).values.flatten(),
        batch_size=2000, epochs=500, verbose=False)
    feature_names = [x for x in dataset.feature_names if x != dataset.col_to_encode]
    train_features_np = np.concatenate([
        dataset.train_features[feature_names],
        encoder.predict(col_to_encode).reshape((-1, n_classes))
    ], axis=1)
    model.train(train_features_np, dataset.train_labels)
    test_features_np = np.concatenate([
        dataset.test_features[feature_names].to_numpy(),
        encoder.predict(dataset.test_features[[dataset.col_to_encode]].values).reshape(-1, n_classes)
    ], axis=1)
    return model.predict(test_features_np)


if __name__ == "__main__":
    args = parser.parse_args()
    for i in range(args.nruns):
        dataset = load_openml_dataset(41216, "Publisher", use_label_encoding=True)
        n_classes = int(dataset.train_labels.nunique())
        preds_dict = {
            "Frequency Encoding": frequency_encoder_model_preds(dataset, MultiClassifier()),
            "Impact Encoding": impact_encoder_model_preds(
                dataset, MultiClassifier(), multiclass=True, num_classes=n_classes),
            "Ordinal Encoding": ordinal_encoder_model_preds(dataset, MultiClassifier()),
            "One Hot Encoding": one_hot_encoder_model_preds(dataset, MultiClassifier()),
            "GLMM Encoding": glmm_encoder_model_preds(dataset, MultiClassifier())
        }

        encoding_types = []
        auc_values = []
        for encoding_type, preds in preds_dict.items():
            encoding_types.append(encoding_type)
            auc_values.append(roc_auc_score(dataset.test_labels, preds, multi_class='ovr'))
        df = pd.DataFrame({"Encoding": encoding_types, "AUNU (one vs. rest AUC-ROC)": auc_values})
        if args.outpath:
            df.to_csv(args.outpath, index=False, header=(i == 0), mode=("w" if i == 0 else "a"))
        else:
            print(df)
