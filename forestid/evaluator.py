import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def calculate_metrics(df: pd.DataFrame):
    report = classification_report(df["gt"], df["prediction"])
    conf_matrix = confusion_matrix(df["gt"], df["prediction"])
    return report, conf_matrix
