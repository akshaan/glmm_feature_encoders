"""Data visualization functions"""
from typing import List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def target_violin_plot_by_feature(data: pd.DataFrame, feature_col: str = "x", target_col: str = "y"):
    sns.violinplot(x=data[feature_col], y=data[target_col], data=data)
    plt.show()


def log_likelihood_loss_plot(losses: List[float]):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(losses, 'k-')
    ax.set(xlabel="Iteration",
           ylabel="Loss (ELBO)",
           title="Loss during training",
           ylim=0)
    plt.show()
