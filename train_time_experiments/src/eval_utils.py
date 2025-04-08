import numpy as np
from sklearn.metrics import roc_curve


def calculate_recall_at_fpr(scores_list, labels_list, target_fpr=0.005):
    """
    Calculate recall at a specific false positive rate for a list of scores and labels.

    Parameters:
    scores_list: List of anomaly scores
    labels_list: List of binary labels (0 for normal, 1 for anomaly)
    target_fpr: Target false positive rate (default 0.005 for 0.5%)

    Returns:
    float: Recall (true positive rate) at the specified FPR
    """
    # Convert lists to numpy arrays
    scores = np.array(scores_list)
    labels = np.array(labels_list)

    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)

    # Find the index where FPR is closest to target_fpr
    idx = np.argmin(np.abs(fpr - target_fpr))
    print(f'target: {target_fpr} - actual: {fpr[idx]}')

    # Return the corresponding recall (TPR)
    return tpr[idx]


def calculate_recalls_for_dataframe(df, target_fpr=0.005, layer = None):
    """
    Calculate recall at 0.5% FPR for each row in the dataframe.

    Parameters:
    df: DataFrame with 'Scores' and 'Labels' columns containing lists

    Returns:
    Series: Recalls at target_fpr for each row
    """
    score_column = 'Scores' if layer is None else f'Layer {layer}.input_layernorm.input Scores'

    recalls = df.apply(
        lambda row: calculate_recall_at_fpr(
            row[score_column],
            row['Labels'],
            target_fpr=target_fpr
        ),
        axis=1
    )
    return recalls
