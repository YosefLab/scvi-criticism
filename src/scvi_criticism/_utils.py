import numpy as np


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
