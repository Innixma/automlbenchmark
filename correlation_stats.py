import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

def heatmap_plot(result):
    fig = plt.figure(figsize=(9, 9), dpi=100)
    ax = fig.add_subplot()
    cax = ax.matshow(result.corr(), interpolation='nearest')
    fig.colorbar(cax)

    num_vars = len(result.corr().columns)
    ax.set_xticks(list(range(num_vars)))
    ax.set_yticks(list(range(num_vars)))
    ax.set_xticklabels(list(result.corr().columns))
    ax.set_yticklabels(list(result.corr().columns))
    ax.tick_params(axis='x', which='major', labelsize=5)
    ax.tick_params(axis='y', which='major', labelsize=5)

    for i in range(num_vars):
        for j in range(num_vars):
            text = ax.text(j, i, round(np.array(result.corr())[i, j], 2),
                           ha="center", va="center", color="w")

    plt.show()

def regression_diff_metric_correlation(vanilla_df: pd.DataFrame, pseudo_df: pd.DataFrame):
    metric = "result"
    vanilla_regression_df = vanilla_df[vanilla_df.type == "regression"]
    pseudo_regression_df = pseudo_df[pseudo_df.type == "regression"]

    score_diff_list, vanilla_score_list, pseudo_score_list, task_list, fold_list = list(), list(), list(), list(), list()

    for task in vanilla_regression_df['task'].unique():
        vanilla_rows = vanilla_regression_df[vanilla_regression_df["task"] == task]
        pseudo_rows = pseudo_regression_df[pseudo_regression_df["task"] == task]

        for fold in vanilla_regression_df["fold"].unique():
            vanilla_row_fold, pseudo_row_fold = vanilla_rows[vanilla_rows["fold"] == fold], pseudo_rows[
                pseudo_rows["fold"] == fold]
            if len(vanilla_row_fold) == 0 or len(pseudo_row_fold) == 0 or vanilla_row_fold[metric].isna().item() or \
                    pseudo_row_fold[metric].isna().item():
                continue
            vanilla_score, pseudo_score = vanilla_row_fold[metric].item(), pseudo_row_fold[metric].item()
            score_dff = pseudo_score - vanilla_score

            score_diff_list.append(score_dff)
            vanilla_score_list.append(vanilla_score)
            pseudo_score_list.append(pseudo_score)
            task_list.append(task)
            fold_list.append(fold)

    df = pd.DataFrame.from_dict(dict(task=task_list, fold=fold_list, vanilla_score=vanilla_score_list, pseudo_score=pseudo_score_list, difference=score_diff_list))
    heatmap_plot(df.corr())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Config')
    parser.add_argument("--config", name="?", nargs="?", type=str, default="correlation_stats.yaml",
                        help="configuration for correlation stats")
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    pseudo_label_path = cfg['pseudo_label_path']
    vanilla_path = cfg['vanilla_path']

    vanilla_df = pd.read_csv(vanilla_path)
    pseudo_df = pd.read_csv(pseudo_label_path)
