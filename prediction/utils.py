import pandas as pd
from datetime import datetime
from loguru import logger
from datasets import load_metric


# change timestamp to datetime object (e.g. 2021-01-07 05:28:00+00:00)
def format_date(datestr: str) -> datetime:
    # remove anything that comes after . or +
    datestr = datestr.split(".")[0]
    datestr = datestr.split("+")[0]
    return datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S")


def load_csv_as_df(fp: str) -> pd.DataFrame:
    return pd.read_csv(fp)


def test_key_agreement_before_merging(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    keys: list[str] = ["source", "entry_id", "version_x", "version_y"],
) -> None:
    # check that the resulting number of groupings are the same
    grouped_df1 = len(df1.groupby(keys))
    grouped_df2 = len(df2.groupby(keys))
    logger.info(
        f"Number of article versions: {grouped_df1}\nNumber of article versions with timestamp: {grouped_df2}"
    )

    assert (
        grouped_df1 == grouped_df2
    ), f"Number of groupings for df1 ({grouped_df1}) and number of groupings for df2 ({grouped_df2}) are not the same"

    # check whether the set of values for keys are the same for both dfs
    df1_values = df1[keys].values
    df1_values = set([tuple(row) for row in df1_values])
    df2_values = df2[keys].values
    df2_values = set([tuple(row) for row in df2_values])
    are_sets_equal = df1_values == df2_values
    assert are_sets_equal, f"Sets of values for {keys} are not the same for both dfs"


def compute_metrics(pred):
    metric = load_metric("accuracy")

    # Get predicted token IDs and labels
    predictions_ids = pred.predictions_ids
    label_ids = pred.label_ids

    # Flatten token IDs
    predictions_flat = [item for sublist in predictions_ids for item in sublist]
    labels_flat = [item for sublist in label_ids for item in sublist]

    # Compute accuracy
    accuracy = metric.compute(predictions=predictions_flat, references=labels_flat)

    # Compute F1 score
    true_positives = sum(
        pred == label for pred, label in zip(predictions_flat, labels_flat)
    )
    false_positives = sum(
        pred != label for pred, label in zip(predictions_flat, labels_flat)
    )

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (
        true_positives + sum(label == 1 for label in labels_flat) + 1e-10
    )

    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1,
    }


def create_finegrained_to_coarsegrained_map(
    coarsegrained_to_finegrained, label_type
) -> dict:
    if label_type == "fact":
        # drop the style entry in COARSEGRAINED_TO_FINEGRAINED
        coarsegrained_to_finegrained.pop("style")
        # add background into fact
        background = coarsegrained_to_finegrained.pop("background")
        coarsegrained_to_finegrained["fact"] += background

    if label_type == "strict_fact":
        # pop all other than fact
        coarsegrained_to_finegrained.pop("style")
        coarsegrained_to_finegrained.pop("background")

    if label_type == "all":
        background = coarsegrained_to_finegrained.pop("background")
        coarsegrained_to_finegrained["fact"] += background

    finegrained_to_coarsegrained = {
        label.lower(): k.lower()
        for k, v in coarsegrained_to_finegrained.items()
        for label in v
    }

    return finegrained_to_coarsegrained


COARSEGRAINED_TO_FINEGRAINED = {
    "style": [
        "Simplification",
        "Emphasize a Point",
        "Syntax Correction",
        "Define term",
        "Delete Unwanted Tone",
        "Style-Guide Adherence",
        "Sensitivity Consideration",
        "Tonal Edits",
        "De-emphasize a Point",
        "Tonal Improvement",
        "Style-Guide Edits",
        "Emphasize importance",
        "De-emphasize importance",
    ],
    "fact": [
        "Update Eye-witness account",
        "Add Additional Sourcing",
        "Event Update",
        "Delete Source-Document Reference",
        "Event Addition",
        "Source-Document Addition",
        "Delete Quote",
        "Quote Update",
        "Additional Sourcing",
        "Add Correction",
        "Quote Addition",
        "Add Eye-witness account",
        "Correction",
        "Delete Event Reference",
        "Source-Document Update",
        "Add Information (Other)",
        "“Update Background",
        "“Add Analysis",
        "(Additional Sourcing",
    ],
    "background": [
        "Update Analysis",
        "Add Analysis",
        "Delete Background",
        "Update Background",
        "Delete Analysis",
        "Add Background",
    ],
}
