from glmm_encoder.datasets.datasets import load_road_safety_dataset
from xgboost import XGBClassifier

from glmm_encoder.examples.example_helpers import (
    frequency_encoder_model_preds,
    impact_encoder_model_preds,
    ordinal_encoder_model_preds,
    glmm_encoder_model_preds,
    one_hot_encoder_model_preds,
    print_scores
)


def train_model_fn(x, y):
    model = XGBClassifier(n_estimators=10).fit(x, y)
    return model


if __name__ == "__main__":
    dataset = load_road_safety_dataset()
    preds_dict = {
        "Frequency Encoding": frequency_encoder_model_preds(dataset, train_model_fn),
        "Impact Encoding": impact_encoder_model_preds(dataset, train_model_fn),
        "Ordinal Encoding": ordinal_encoder_model_preds(dataset, train_model_fn),
        "One Hot Encoding": one_hot_encoder_model_preds(dataset, train_model_fn),
        "GLMM Encoding": glmm_encoder_model_preds(dataset, train_model_fn),
    }
    print_scores(dataset, preds_dict)
