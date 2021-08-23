from typing import List

from glmm_encoder.examples.dataset_utils import Dataset, load_openml_dataset
from glmm_encoder.encoders import GLMMRegressionFeatureEncoder
from xgboost import XGBRegressor
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import math
import argparse

from glmm_encoder.examples.model_utils import (
    frequency_encoder_model_preds,
    impact_encoder_model_preds,
    ordinal_encoder_model_preds,
    one_hot_encoder_model_preds,
    Model
)

parser = argparse.ArgumentParser(description='Avocado Sales Regression Task')
parser.add_argument("--nruns", type=int, help="number of runs", default=1)
parser.add_argument("--outpath", type=str, help="output path for scores from model runs", default=None)


class Regressor(Model):
    def __init__(self):
        self.xgb_model = XGBRegressor(n_estimators=10, verbosity=0)

    def train(self, train_x: np.array, train_y: np.array) -> None:
        self.xgb_model.fit(train_x, train_y)

    def predict(self, test_x: np.array) -> np.array:
        return self.xgb_model.predict(test_x)


def glmm_encoder_model_preds(dataset: Dataset, model: Model) -> List[float]:
    num_levels = int(dataset.train_features[[dataset.col_to_encode]].nunique())
    col_to_encode = dataset.train_features[[dataset.col_to_encode]].values.flatten()
    encoder = GLMMRegressionFeatureEncoder(num_levels)
    encoder.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-2))
    encoder.fit(
        col_to_encode,
        dataset.train_labels.astype(np.float32).values.flatten(),
        batch_size=2000, epochs=50, verbose=False)
    feature_names = [x for x in dataset.feature_names if x != dataset.col_to_encode]
    train_features_np = np.concatenate([
        dataset.train_features[feature_names],
        encoder.predict(col_to_encode)
    ], axis=1)
    model.train(train_features_np, dataset.train_labels)
    test_features_np = np.concatenate([
        dataset.test_features[feature_names].to_numpy(),
        encoder.predict(dataset.test_features[[dataset.col_to_encode]].values)
    ], axis=1)

    return model.predict(test_features_np)


if __name__ == "__main__":
    args = parser.parse_args()
    for i in range(args.nruns):
        dataset = load_openml_dataset(41210, "region")
        n_levels = dataset.train_features[[dataset.col_to_encode]].nunique()
        preds_dict = {
            "Frequency Encoding": frequency_encoder_model_preds(dataset, Regressor()),
            "Impact Encoding": impact_encoder_model_preds(dataset, Regressor()),
            "Ordinal Encoding": ordinal_encoder_model_preds(dataset, Regressor()),
            "One Hot Encoding": one_hot_encoder_model_preds(dataset, Regressor()),
            "GLMM Encoding": glmm_encoder_model_preds(dataset, Regressor()),
        }

        encoding_types = []
        mse_values = []
        for encoding_type, preds in preds_dict.items():
            encoding_types.append(encoding_type)
            mse_values.append(math.sqrt((mean_squared_error(dataset.test_labels, preds))))
        df = pd.DataFrame({"Encoding": encoding_types, "RMSE": mse_values})
        if args.outpath:
            df.to_csv(args.outpath, index=False, header=(i == 0), mode=("w" if i == 0 else "a"))
        else:
            print(df)
