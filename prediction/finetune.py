# Load model directly
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback,
)
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from seq2seq_dataset import Seq2SeqDataset, create_dataloader
from utils import compute_metrics
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    dropout_rate: float = field(default=0.1, metadata={"help": "Dropout rate."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    prediction_method: str = field(
        default="target_sentence_only",
        metadata={
            "help": "One of [target_sentence_only, target_sentence_and_context]. Default: target_sentence_only"
        },
    )


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # prepare the models
    tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384")

    df = pd.read_csv(
        "/Users/jcho/projects/NewsEdits/data/prediction_data/v2-silver-labeled-data-for-prediction-with-date-and-split.csv"
    )

    if data_args.prediction_method == "target_sentence_only":
        # remove cases that have no target sentence
        data = data[~data["sentence"].isna()]

    target_column = "update_category"
    seq2seq_dataset = Seq2SeqDataset(df, tokenizer, target="target_sentence_only")
    train_dataset = seq2seq_dataset.create_dataset("train")
    eval_dataset = seq2seq_dataset.create_dataset("dev")
    test_dataset = seq2seq_dataset.create_dataset("test")

    # prepare the data
    # Example usage:
    # Load your DataFrame
    # df = pd.read_csv("your_data.csv")

    # Define your target column
    # target_column = "your_target_column"

    # Define the split (train/dev/test)
    # split = "train"

    # Load a pre-trained tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("your_model_name")

    # Create the dataset
    # seq2seq_dataset = Seq2SeqDataset(df, tokenizer)
    # train_dataset = seq2seq_dataset.create_dataset(split)

    # define the data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        "longformer-trainer",
        evaluation_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        eval_steps=500,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # evaluate on test set
    results = trainer.evaluate(test_dataset)
    print(results)


if __name__ == "__main__":
    main()
