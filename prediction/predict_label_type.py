# predict update category for a target sentence using OpenAI models
# saves results to csv file
# usage: python predict_label_type.py --model_name gpt-3.5-turbo --prompt_type sentence_only --test --pred_fp predictions.csv
# usage for gold set: python predict_label_type.py --model_name gpt-3.5-turbo --prompt_type sentence_only --gold --pred_fp predictions.csv

from openai import OpenAI
from utils import load_csv_as_df
import numpy as np
from loguru import logger
from definitions import DETAILED_LABEL_DEFINITIONS, HIGH_LEVEL_LABEL_DEFINITIONS
from argparse import ArgumentParser
from pprint import pprint
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
import json
from collections import Counter
import pandas as pd


class OpenAIClassifier:
    def __init__(self, model_name: str, label_type: str):

        self.client = OpenAI()
        self.model_name = model_name
        self.label_type = label_type

        self.detailed_label_definitions = DETAILED_LABEL_DEFINITIONS
        self.high_level_label_definitions = HIGH_LEVEL_LABEL_DEFINITIONS

        self._format_label_descriptions()

    def predict_label_with_target_sentence_only(
        self, target_sentence: str, show_input: bool = False
    ) -> str:

        prompt = self._form_target_sentence_only_prompt(target_sentence)
        messages = self._form_messages(prompt)

        if show_input:
            logger.info(f"Messages:")
            pprint(messages)

        return self.predict(messages)

    def predict(self, messages) -> str:

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
        )

        return chat_completion.choices[0].message.content

    def _format_label_descriptions(self) -> None:

        self.high_level_label_description = ""
        for label, definition in self.high_level_label_definitions.items():
            if self.label_type == "fact":
                if label in ["style"]:
                    continue
            self.high_level_label_description += f"\n\t{label.lower()}: {definition}\n"

        self.detailed_label_description = ""
        for label, definition in self.detailed_label_definitions.items():
            self.detailed_label_description += f"\n\t{label.lower()}: {definition}\n"

    def form_system_prompt(self) -> str:

        system_prompt = f"Predict which type of update that the given target sentence will have given these label descriptions and context, if available.\n Labels:{self.high_level_label_description}\n"

        return system_prompt

    def _form_target_sentence_only_prompt(self, sample) -> str:

        prompt = f"Target sentence: {sample['sentence']}\nLabel:"

        return prompt

    def _form_target_with_direct_context_prompt(self, sample) -> str:

        prompt = f"Context:{sample['direct_context']}\nTarget sentence: {sample['sentence']}\nLabel:"

        return prompt

    def _form_target_with_full_article_prompt(self, sample) -> str:

        prompt = f"Article:{sample['full_article']}\nTarget sentence: {sample['sentence']}\nLabel:"

        return prompt

    def _form_messages(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        return messages


def main():

    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to use for prediction. Default: gpt-3.5-turbo. One of [gpt-3.5-turbo, gpt-4-0125-preview, or other custom trained models]",
        default="gpt-3.5-turbo",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        help="Type of prompt to use for prediction. One of [sentence_only, direct_context, full_article]. Default: sentence_only",
        default="sentence_only",
    )
    parser.add_argument(
        "--label_type",
        type=str,
        default="fact",
        help="Type of label to focus on. One of ['fact', 'all']. All looks at 'style' labels as well.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use small sample of test data for testing the script",
    )
    parser.add_argument(
        "--gold",
        action="store_true",
        help="Use gold labels for testing the script",
    )
    parser.add_argument(
        "--pred_fp",
        type=str,
        help="Path to save predictions. args parameters will automatically be added to file name to distinguish runs. Default: predictions/predictions.csv",
        default="predictions/predictions.csv",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to data directory where outputs from `gpt_finetuning.py --format_data` are stored",
        default="/project/jonmay_231/hjcho/NewsEdits/prediction/openai_data/",
    )
    parser.add_argument(
        "--repo_dir",
        type=str,
        help="Path to repository directory",
        default="/project/jonmay_231/hjcho/NewsEdits",
    )

    args = parser.parse_args()

    # load files from result of gpt_finetuning.py, already formatted as messages
    load_test_fp = (
        Path(args.data_dir) / f"{args.prompt_type}_{args.label_type}_test.jsonl"
    )
    if args.gold:
        logger.info("Using gold labels for testing...")
        load_test_fp = (
            Path(args.data_dir)
            / f"{args.prompt_type}_{args.label_type}_gold_test.jsonl"
        )

    # load original test data with all columns to add back finegrained labels
    load_test_fp_all_columns = (
        Path(args.repo_dir)
        / f"data/prediction_data/v3-silver-labeled-data-for-prediction-with-date-test_2000.csv"
    )
    if args.gold:
        load_test_fp_all_columns = (
            Path(args.repo_dir)
            / f"data/prediction_data/gold_test_v2_{args.label_type}.csv"
        )
    df_all_columns = load_csv_as_df(load_test_fp_all_columns)

    with open(load_test_fp, "r") as f:
        data = [json.loads(line) for line in f]

    # sample to only use args.testset_size
    if args.test:
        testset_size = 5
        logger.info(f"Sampling {testset_size} from test set for testing")
        data = data[:testset_size]
        df_all_columns = df_all_columns[:testset_size]

    # calculate label distribution
    labels = [i["messages"][-1]["content"] for i in data]

    counter = Counter(labels)
    logger.info(counter)

    # initialize classifier
    openai_classifier = OpenAIClassifier(
        model_name=args.model_name, label_type=args.label_type
    )

    # predict label for test set
    predictions = []
    for sample in tqdm(data):
        input_messages = sample["messages"][:-1]  # last message contains the label
        prediction = openai_classifier.predict(input_messages)
        predictions.append(prediction.lower().strip())

        if args.test:
            logger.info(f"Sample input: {input_messages}")
            logger.info(f"Prediction: {prediction}")
            logger.info(f"True label: {sample['messages'][-1]['content']}")

    accuracy = (
        sum([1 for i, j in zip(labels, predictions) if i == j]) / len(predictions) * 100
    )
    logger.info(f"Overall accuracy: {accuracy:.2f}\%")

    # get f1 score
    f1 = f1_score(labels, predictions, average="weighted")
    logger.info(f"Overall F1 score: {f1:.2f}")

    # print f1 score for each update category ("fact", "style", "none")
    update_categories = sorted(list(set(labels)))
    for category in update_categories:
        indices = [i for i, j in enumerate(labels) if j == category]
        f1 = f1_score(
            [labels[i] for i in indices],
            [predictions[i] for i in indices],
            average="weighted",
        )
        logger.info(f"F1 score for {category}: {f1:.2f}")

    if args.test:
        return

    print(classification_report(labels, predictions))

    # save predictions
    df_dict = {
        "sentence": [i["messages"][1]["content"] for i in data],
        "update_category": labels,
        "prediction": predictions,
        "labels": df_all_columns["label"],  # track finegrained labels as well
    }

    # add model name and number of test samples to filepath
    pred_fp = args.pred_fp.replace(
        ".csv",
        f"_{args.prompt_type}_{args.label_type}_{args.model_name}_{len(predictions)}.csv",
    )

    df = pd.DataFrame(df_dict)
    df.to_csv(pred_fp, index=False)

    return


if __name__ == "__main__":
    main()
