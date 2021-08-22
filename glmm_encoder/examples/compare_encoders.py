import argparse

import os
import inspect
from pathlib import Path
import pandas as pd
import seaborn as sns
import tempfile
import shutil
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Video Genre Multiclass Classification Task')
parser.add_argument("--scores_dir", type=str, help="dir where eval scores should be written", default=None)
parser.add_argument("--nruns", type=int, help="number of runs per task", default=100)
parser.add_argument("--plot_only", action="store_true", help="Plot values at --scores_dir without running any evals")


def target_violin_plot_by_feature(data: pd.DataFrame, y_col, ax, title, invert_y=False):
    plot = sns.violinplot(x=data["Encoding"], y=data[y_col], data=data, ax=ax)
    plot.set_title(title)
    plot.set(xlabel=None)
    if invert_y:
        ax.invert_yaxis()


if __name__ == "__main__":
    args = parser.parse_args(["--scores_dir", "/Users/akshaankakar/Desktop/glmm_runs/"])

    # Run all tasks to get output scores
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    examples_dir = Path(filename).resolve().parent
    scores_dir = args.scores_dir if args.scores_dir else tempfile.mkdtemp()
    if not args.plot_only:
        for task in [
            "regression/avocado_sales_example.py",
            "multiclass_classification/video_game_sales_example.py",
            "binary_classification/churn_example.py"]:
            print(f"Running task examples/{task}, Num runs (train/test splits) = {args.nruns}")
            script_path = str(examples_dir / task)
            output_path = str(Path(scores_dir) / (task.split("/")[0] + "_scores.csv"))
            script_args = f"--nruns {args.nruns} --outpath {output_path}"
            os.system(f"python -W ignore {script_path} {script_args}")

    # Plot output scores
    df_regression = pd.read_csv(Path(scores_dir) / "regression_scores.csv")
    df_binary = pd.read_csv(Path(scores_dir) / "binary_classification_scores.csv")
    df_multiclass = pd.read_csv(Path(scores_dir) / "multiclass_classification_scores.csv")
    regression_title = "RMSE for Avocado Sales Regression\n(OpenML Id = 41210 , Encoded feature = 'region')"
    binary_title = "ROC-AUC for Churn Binary Classification (OpenML Id = 41210, Encoded feature = 'region')"
    multiclass_title = ("One vs. Rest ROC-AUC (AUNU) for Video Game Genre Classification (OpenML Id = 41216, "
                        "Encoded feature = 'Publisher')")

    figure, axis = plt.subplots(3, 1)
    target_violin_plot_by_feature(df_regression, "RMSE", axis[0], regression_title)
    target_violin_plot_by_feature(df_binary, "AUC-ROC", axis[1], binary_title, invert_y=True)
    target_violin_plot_by_feature(df_multiclass, "AUNU (one vs. rest AUC-ROC)", axis[2], multiclass_title,
                                  invert_y=True)

    figure.suptitle(f"Comparison of Encoding Types Across Prediction Tasks (Num. runs per task = {args.nruns})",
                    fontsize=16)
    plt.tight_layout()
    plt.show()

    # Clean up tempdir if one was used
    if not scores_dir:
        shutil.rmtree(scores_dir)
