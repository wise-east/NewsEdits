import pandas as pd
from transformers import AutoTokenizer
from predict_label_type import OpenAIClassifier
from datasets import load_dataset, Dataset
from argparse import ArgumentParser
import numpy as np
from openai import OpenAI
import json
from loguru import logger
from utils import COARSEGRAINED_TO_FINEGRAINED
from pathlib import Path


class GPTFinetuningManager:
    def __init__(self, model_name: str = "gpt-3.5-turbo", args=None):
        self.args = args
        self.data_version = args.data_version
        self.label_type = args.label_type
        self.seed = self.args.seed
        self.train_size = self.args.train_size
        self.validation_size = self.args.validation_size
        self.save_dir = self.args.save_dir

        self.classifier = OpenAIClassifier(model_name, self.label_type)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        assert self.label_type in [
            "fact",
            "strict_fact",
            "all",
        ], f"label type: {self.label_type} not in ['fact', 'strict_fact', 'all']"

        self.home_dir = args.home_dir

        self.load_data()

    def load_data(self):

        # load data
        self.train_data = pd.read_csv(
            f"{self.home_dir}/data/prediction_data/v{self.data_version}-silver-labeled-data-for-prediction-with-date-train_{self.label_type}.csv"
        )
        self.validation_data = pd.read_csv(
            f"{self.home_dir}/data/prediction_data/v{self.data_version}-silver-labeled-data-for-prediction-with-date-dev_{self.label_type}.csv"
        )
        self.test_data = pd.read_csv(
            f"{self.home_dir}/data/prediction_data/v{self.data_version}-silver-labeled-data-for-prediction-with-date-test_{self.label_type}.csv"
        )

        self.gold_test_data = pd.read_csv(
            f"{self.home_dir}/data/prediction_data/gold_test_v2_{self.label_type}.csv"
        )

    def sample_from_data(self):

        self.train_dataset = Dataset.from_pandas(self.train_data, split="train")
        self.validation_dataset = Dataset.from_pandas(
            self.validation_data, split="validation"
        )
        self.test_dataset = Dataset.from_pandas(self.test_data, split="test")
        self.gold_test_dataset = Dataset.from_pandas(self.gold_test_data, split="test")

        # shuffle with seed to 42  and sample 16000
        self.train_dataset = self.train_dataset.shuffle(seed=self.seed).select(
            range(self.train_size)
        )
        # do the same with 2000 for validation and test
        self.validation_dataset = self.validation_dataset.shuffle(
            seed=self.seed
        ).select(range(self.validation_size))
        self.test_dataset = self.test_dataset.shuffle(seed=self.seed).select(
            range(self.validation_size)
        )

    def save(self):
        # save datasets to csv
        self.train_dataset.to_csv(
            f"{self.home_dir}/data/prediction_data/v{self.data_version}-silver-labeled-data-for-prediction-with-date-train_{self.label_type}_{self.train_size}.csv"
        )
        self.validation_dataset.to_csv(
            f"{self.home_dir}/data/prediction_data/v{self.data_version}-silver-labeled-data-for-prediction-with-date-dev_{self.label_type}_{self.validation_size}.csv"
        )
        self.test_dataset.to_csv(
            f"{self.home_dir}/data/prediction_data/v{self.data_version}-silver-labeled-data-for-prediction-with-date-test_{self.label_type}_{self.validation_size}.csv"
        )

    def calc_finetuning_total_costs(self):

        for dataset in [self.train_dataset, self.validation_dataset, self.test_dataset]:
            dataset = dataset.map(
                lambda x: {
                    "sentence_only_prompt": self.classifier._form_target_sentence_only_prompt(
                        x
                    )
                    + x["update_category"],
                    "direct_context_prompt": self.classifier._form_target_with_direct_context_prompt(
                        x
                    )
                    + x["update_category"],
                    "full_article_prompt": self.classifier._form_target_with_full_article_prompt(
                        x
                    )
                    + x["update_category"],
                }
            )

        # get total token counts for each prompt
        sentence_only_train_total = np.sum(
            [
                len(x)
                for x in self.tokenizer(train_dataset["sentence_only_prompt"])[
                    "input_ids"
                ]
            ]
        )
        direct_context_train_total = np.sum(
            [
                len(x)
                for x in self.tokenizer(train_dataset["direct_context_prompt"])[
                    "input_ids"
                ]
            ]
        )
        full_article_train_total = np.sum(
            [
                len(x)
                for x in self.tokenizer(train_dataset["full_article_prompt"])[
                    "input_ids"
                ]
            ]
        )

        sentence_only_validation_total = np.sum(
            [
                len(x)
                for x in self.tokenizer(validation_dataset["sentence_only_prompt"])[
                    "input_ids"
                ]
            ]
        )
        direct_context_validation_total = np.sum(
            [
                len(x)
                for x in self.tokenizer(validation_dataset["direct_context_prompt"])[
                    "input_ids"
                ]
            ]
        )
        full_article_validation_total = np.sum(
            [
                len(x)
                for x in self.tokenizer(validation_dataset["full_article_prompt"])[
                    "input_ids"
                ]
            ]
        )

        sentence_only_test_total = np.sum(
            [
                len(x)
                for x in self.tokenizer(test_dataset["sentence_only_prompt"])[
                    "input_ids"
                ]
            ]
        )
        direct_context_test_total = np.sum(
            [
                len(x)
                for x in self.tokenizer(test_dataset["direct_context_prompt"])[
                    "input_ids"
                ]
            ]
        )
        full_article_test_total = np.sum(
            [
                len(x)
                for x in self.tokenizer(test_dataset["full_article_prompt"])[
                    "input_ids"
                ]
            ]
        )

        counts = [
            {
                "type": "sentence_only",
                "train": sentence_only_train_total,
                "validation": sentence_only_validation_total,
                "test": sentence_only_test_total,
            },
            {
                "type": "direct_context",
                "train": direct_context_train_total,
                "validation": direct_context_validation_total,
                "test": direct_context_test_total,
            },
            {
                "type": "full_article",
                "train": full_article_train_total,
                "validation": full_article_validation_total,
                "test": full_article_test_total,
            },
        ]

        # print out the results
        df = pd.DataFrame(counts)
        # set the type as index
        df.set_index("type", inplace=True)

        # sum column
        df["total"] = df[["train", "validation", "test"]].sum(
            axis=1
        )  # sum train and validation columns
        df["train_validation_total"] = df["train"] + df["validation"]

        # compute costs
        cost_per_1000_tokens_finetuning = 0.008
        cost_per_1000_tokens_inference_finetuned = 0.003
        cost_per_1000_tokens_inference_gpt35 = 0.0005
        cost_per_1000_tokens_inference_gpt4 = 0.01
        num_epochs = 3

        df["training_cost"] = (
            (df["train"] + df["validation"])
            * cost_per_1000_tokens_finetuning
            / 1000
            * num_epochs
        )
        df["inference_cost"] = (
            df["test"] * cost_per_1000_tokens_inference_finetuned / 1000
        )
        df["total_cost"] = df["training_cost"] + df["inference_cost"]

        df["0shot_cost_gpt3.5"] = (
            df["test"] * cost_per_1000_tokens_inference_gpt35 / 1000
        )
        df["0shot_cost_gpt4"] = df["test"] * cost_per_1000_tokens_inference_gpt4 / 1000

        print(df)

    def format_all_data_and_save_for_gpt_finetuning(self) -> str:
        # convenience function for formatting and saving all data into gpt messages format

        validation_fps = self.format_and_save_data_for_gpt_finetuning(
            self.validation_dataset, "validation", save_dir=self.save_dir
        )
        test_fps = self.format_and_save_data_for_gpt_finetuning(
            self.test_dataset, "test", save_dir=self.save_dir
        )
        fps = self.format_and_save_data_for_gpt_finetuning(
            self.train_dataset, "train", save_dir=self.save_dir
        )

        gold_fps = self.format_and_save_data_for_gpt_finetuning(
            self.gold_test_dataset, "gold_test", save_dir=self.save_dir
        )

        all_fps = {**fps, **validation_fps, **test_fps, **gold_fps}

        logger.info(f"Files saved for finetuning: {fps}")
        return all_fps

    def format_and_save_data_for_gpt_finetuning(
        self, dataset: Dataset, split: str, save_dir: str
    ) -> str:

        if not Path(save_dir).exists():
            Path(save_dir).mkdir()

        dataset = dataset.map(
            lambda x: {
                "sentence_only_messages": {
                    "messages": [
                        {
                            "role": "system",
                            "content": self.classifier.form_system_prompt(),
                        },
                        {
                            "role": "user",
                            "content": self.classifier._form_target_sentence_only_prompt(
                                x
                            ),
                        },
                        {"role": "assistant", "content": x["update_category"]},
                    ],
                },
                "direct_context_messages": {
                    "messages": [
                        {
                            "role": "system",
                            "content": self.classifier.form_system_prompt(),
                        },
                        {
                            "role": "user",
                            "content": self.classifier._form_target_with_direct_context_prompt(
                                x
                            ),
                        },
                        {"role": "assistant", "content": x["update_category"]},
                    ]
                },
                "full_article_messages": {
                    "messages": [
                        {
                            "role": "system",
                            "content": self.classifier.form_system_prompt(),
                        },
                        {
                            "role": "user",
                            "content": self.classifier._form_target_with_full_article_prompt(
                                x
                            ),
                        },
                        {"role": "assistant", "content": x["update_category"]},
                    ]
                },
            }
        )

        fps = {}
        # get list of messages for each prompt type
        for prompt_type in ["sentence_only", "direct_context", "full_article"]:
            messages_list = dataset[prompt_type + "_messages"]

            fp = Path(save_dir) / f"{prompt_type}_{self.label_type}_{split}.jsonl"
            with fp.open("w") as f:
                for messages in messages_list:
                    f.write(json.dumps(messages) + "\n")

            fps[f"{prompt_type}_{split}"] = fp

        return fps


def submit_finetuning_job(train_fp: str, suffix: str, model: str = "gpt-3.5-turbo"):

    client = OpenAI()

    # upload file
    upload_result = client.files.create(file=open(train_fp, "rb"), purpose="fine-tune")

    logger.info(upload_result)

    # start finetuning
    finetune_result = client.fine_tuning.jobs.create(
        training_file=upload_result.id,
        model=model,
        suffix=suffix,
        hyperparameters={"n_epochs": 1},
    )

    logger.info(finetune_result)

    client.fine_tuning.jobs.list(limit=10)


def main():

    parser = ArgumentParser()
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="sentence_only",
        help="prompt type to use for finetuning. One of 'sentence_only', 'direct_context', 'full_article'",
    )
    parser.add_argument(
        "--calc_finetuning_total_costs",
        action="store_true",
        help="calculate the total costs for finetuning and inference",
    )
    parser.add_argument(
        "--label_type",
        type=str,
        default="strict_fact",
        help="Type of label to focus on. One of ['fact', 'strict_fact', 'all']. All looks at 'style' labels as well.",
    )
    parser.add_argument(
        "--format_data", action="store_true", help="format and save data for finetuning"
    )
    parser.add_argument(
        "--fp",
        type=str,
        help="file path to use for finetuning, should be a jsonl file with openai messages formatting",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo", help="model to use for finetuning"
    )
    args = parser.parse_args()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="openai_data",
        help="directory to save finetuning files",
    )
    parser.add_argument(
        "--data_version",
        type=int,
        default=3,
        help="version of the data to use for finetuning",
    )
    parser.add_argument(
        "--home_dir",
        type=str,
        default="/project/jonmay_231/hjcho/NewsEdits",
        help="home directory for the project",
    )
    parser.add_argument(
        "--train_size", type=int, default=16000, help="size of the training dataset"
    )
    parser.add_argument(
        "--validation_size",
        type=int,
        default=2000,
        help="size of the validation dataset",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for shuffling the datasets"
    )

    args = parser.parse_args()

    logger.info(args)

    gpt_manager = GPTFinetuningManager(model_name=args.model, args=args)
    gpt_manager.sample_from_data()
    gpt_manager.save()

    if args.calc_finetuning_total_costs:
        gpt_manager.calc_finetuning_total_costs()

    if args.format_data and args.fp is None:
        all_fps = gpt_manager.format_all_data_and_save_for_gpt_finetuning()
        return

    else:
        fp = args.fp

    # assert
    assert (
        args.prompt_type in fp
    ), f"prompt type: {args.prompt_type} not in file path: {fp}, check that the file path is correct or that the prompt type is correct."

    logger.info(
        f"Submitting finetuning job for prompt type: {args.prompt_type} using file: {fp}"
    )

    # submit for finetuning
    submit_finetuning_job(fp, model=model, suffix=args.prompt_type)

    return


if __name__ == "__main__":
    main()
