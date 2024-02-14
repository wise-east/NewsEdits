import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from loguru import logger


class Seq2SeqDataset:
    def __init__(
        self,
        dataframe,
        tokenizer,
        max_input_length=512,
        max_output_length=128,
        **kwargs,
    ):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.args = kwargs

    def format_input_example(self, examples):

        breakpoint()

        if self.args.get("target") == "target_sentence_only":
            # prepend "target sentence:" and append "label:" for each example
            try:
                return "target sentence: " + examples["sentence"] + " label: "
            except Exception as e:
                logger.error(f"Error in format_input_example for example: {examples}")
                breakpoint()
                raise e
        else:
            raise NotImplementedError(
                f"Not implemented yet for {self.args.get('target')}\nOnly target_sentence_only is supported at the moment."
            )

    def process_data(self, examples):

        inputs = self.tokenizer(
            # examples["sentence"] + " " + examples["title"],
            self.format_input_example(examples),
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        outputs = self.tokenizer(
            examples["update_category"],
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs["labels"] = outputs["input_ids"]

        return inputs

    def create_dataset(self, split):
        split_data = self.data[self.data["split"] == split]
        dataset = Dataset.from_pandas(split_data)
        return dataset.map(self.process_data)


def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
