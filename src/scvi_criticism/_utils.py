import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# from https://stackoverflow.com/a/28216751
def _add_identity(axes, *line_args, **line_kwargs):
    (identity,) = axes.plot([], [], *line_args, **line_kwargs, label="identity")

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect("xlim_changed", callback)
    axes.callbacks.connect("ylim_changed", callback)
    axes.legend()
    return axes


def _get_dp_as_df(dp, kind: str = "color"):
    fractions = dp.dot_size_df
    colors = dp.dot_color_df
    assert fractions.columns.equals(colors.columns)
    assert fractions.index.equals(colors.index)
    if kind == "color":
        return colors
    elif kind == "fraction":
        return fractions
    else:
        raise ValueError(f"Unknown kind: {kind}")


def _get_df_mae(df_1, df_2):
    # returns a tuple with:
    # 1. the row-wise mean absolute error between the two df's
    # 2. the mean of the mean absolute errors across all rows
    df_diff = pd.DataFrame(df_1 - df_2, dtype="float64")
    mae = np.abs(df_diff).mean(axis=1)
    return mae, np.mean(mae)


def _get_df_corr_coeff(df_1, df_2, corr_coeff: str, remove_dup_cols: bool = True):
    # returns a tuple with:
    # 1. the row-wise correlation coefficient (pearson or spearman) between the two df's
    # 2. the mean of the correlation coefficients across all rows
    assert df_1.index.equals(df_2.index)
    assert df_1.columns.equals(df_2.columns)
    if remove_dup_cols:
        x = df_1.loc[:, ~df_1.columns.duplicated()]
        y = df_2.loc[:, ~df_2.columns.duplicated()]
    cc_df = pd.Series(index=df_1.index, dtype="float64")
    for r in df_1.index:
        if corr_coeff == "pearson":
            cc_df[r] = pearsonr(x.loc[r], y.loc[r])[0]
        elif corr_coeff == "spearman":
            cc_df[r] = spearmanr(x.loc[r], y.loc[r])[0]
        else:
            raise ValueError(f"Unknown corr_coeff option: {corr_coeff}")
    return cc_df, np.mean(cc_df)


def _get_precision_recall_f1(ground_truth: np.ndarray, pred: np.ndarray, do_round: bool = True):
    assert type(ground_truth) == np.ndarray and type(pred) == np.ndarray
    # https://stackoverflow.com/a/68157457
    tp = np.sum(np.logical_and(pred == 1, ground_truth == 1))
    fp = np.sum(np.logical_and(pred == 1, ground_truth == 0))
    fn = np.sum(np.logical_and(pred == 0, ground_truth == 1))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    if do_round:
        dec = 2
        return np.round(precision, dec), np.round(recall, dec), np.round(f1, dec)
    else:
        return precision, recall, f1


def _get_binary_array_from_selection(all_vals, selected_vals) -> np.ndarray:
    b = [0 if val not in selected_vals else 1 for val in all_vals]
    return np.array(b)
