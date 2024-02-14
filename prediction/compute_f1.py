import json
from argparse import ArgumentParser
from sklearn.metrics import f1_score, classification_report
import pandas as pd


def compute_f1(y_true, y_pred, average="macro"):
    return f1_score(y_true, y_pred, average=average)


parser = ArgumentParser()
parser.add_argument("--predictions_fp", type=str, required=True)
parser.add_argument(
    "--gold_fp", type=str, help="Path to the gold labels", required=False
)

args = parser.parse_args()

# if it's a csv file, use pandas to read it
predictions_fp = args.predictions_fp


if predictions_fp.endswith(".csv"):
    df = pd.read_csv(predictions_fp)
    labels = df["update_category"]
    predictions = df["prediction"]

else:
    with open(predictions_fp, "r") as f:
        predictions = f.readlines()

    predictions = [p.replace("\n", "").strip() for p in predictions]

    if args.gold_fp:
        test_fp = args.gold_fp
        # if it ends with csv
        if test_fp.endswith(".csv"):
            df = pd.read_csv(test_fp)
            labels = df["update_category"]
        else:  # if it ends with jsonl
            with open(test_fp, "r") as f:
                test_data = [json.loads(line) for line in f]

            labels = [d["messages"][-1]["content"] for d in test_data]
    else:
        test_fp = "/project/jonmay_231/hjcho/NewsEdits/prediction/openai_data/sentence_only_test.jsonl"

        with open(test_fp, "r") as f:
            test_data = [json.loads(line) for line in f]

        labels = [d["messages"][-1]["content"] for d in test_data]


overall_accuracy = sum(
    [1 if label == prediction else 0 for label, prediction in zip(labels, predictions)]
) / len(labels)
overall_weighted_score = compute_f1(labels, predictions, average="weighted")
overall_macro_score = compute_f1(labels, predictions, average="macro")
overall_micro_score = compute_f1(labels, predictions, average="micro")

print(f"Overall accuracy: {overall_accuracy*100:.2f}%")
print(f"Overall macro F1 score: {overall_macro_score:.2f}")
print(f"Overall micro F1 score: {overall_micro_score:.2f}")
print(f"Overall weighted F1 score: {overall_weighted_score:.2f}")

f1_scores_individual = compute_f1(labels, predictions, average=None)
print("F1 Score per Class:", f1_scores_individual)

# Print the F1 score per update category
category_f1_scores = {}
results_columns = []
for category in ["fact", "style", "none"]:
    category_labels = [1 if label == category else 0 for label in labels]
    category_predictions = [
        1 if prediction == category else 0 for prediction in predictions
    ]

    if len(category_labels):
        continue

    category_accuracy = sum(
        [
            1 if label == category and prediction == category else 0
            for label, prediction in zip(labels, predictions)
        ]
    ) / sum(category_labels)

    # compute f1 score for each class
    category_f1 = compute_f1(category_labels, category_predictions, average="binary")
    print(f"F1 score for {category}: {category_f1:.2f}")
    print(f"Accuracy for {category}: {category_accuracy*100:.2f}%")
    category_f1_scores[category] = category_f1

    results_columns.append(category_f1)

print(classification_report(labels, predictions))

results_columns += [
    overall_macro_score,
    overall_micro_score,
    overall_weighted_score,
]

# overleaf_results_str =
result_str = " & ".join([f"{score:.2f}" for score in results_columns])
print(result_str)
