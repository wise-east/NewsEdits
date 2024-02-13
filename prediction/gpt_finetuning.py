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


def calc_finetuning_total_costs(train_dataset, validation_dataset, test_dataset, classifier, tokenizer): 
    
    for dataset in [train_dataset, validation_dataset, test_dataset]:
        dataset = dataset.map(lambda x : {
            "sentence_only_prompt": classifier._form_target_sentence_only_prompt(x) + x["update_category"],
            "direct_context_prompt": classifier._form_target_with_direct_context_prompt(x) + x["update_category"],
            "full_article_prompt": classifier._form_target_with_full_article_prompt(x) + x["update_category"]
            }
        )

    # get total token counts for each prompt
    sentence_only_train_total = np.sum([len(x) for x in tokenizer(train_dataset["sentence_only_prompt"])["input_ids"]])
    direct_context_train_total = np.sum([len(x) for x in tokenizer(train_dataset["direct_context_prompt"])["input_ids"]])
    full_article_train_total = np.sum([len(x) for x in tokenizer(train_dataset["full_article_prompt"])["input_ids"]])

    sentence_only_validation_total = np.sum([len(x) for x in tokenizer(validation_dataset["sentence_only_prompt"])["input_ids"]])
    direct_context_validation_total = np.sum([len(x) for x in tokenizer(validation_dataset["direct_context_prompt"])["input_ids"]])
    full_article_validation_total = np.sum([len(x) for x in tokenizer(validation_dataset["full_article_prompt"])["input_ids"]])

    sentence_only_test_total = np.sum([len(x) for x in tokenizer(test_dataset["sentence_only_prompt"])["input_ids"]])
    direct_context_test_total = np.sum([len(x) for x in tokenizer(test_dataset["direct_context_prompt"])["input_ids"]])
    full_article_test_total = np.sum([len(x) for x in tokenizer(test_dataset["full_article_prompt"])["input_ids"]])

    counts = [
        {
            "type": "sentence_only",
            "train": sentence_only_train_total,
            "validation": sentence_only_validation_total,
            "test": sentence_only_test_total
        },
        {
            "type": "direct_context",
            "train": direct_context_train_total,
            "validation": direct_context_validation_total,
            "test": direct_context_test_total
        },
        {
            "type": "full_article",
            "train": full_article_train_total,
            "validation": full_article_validation_total,
            "test": full_article_test_total
        }
    ]

    # print out the results
    df = pd.DataFrame(counts)
    # set the type as index 
    df.set_index("type", inplace=True)

    # sum column
    df["total"] = df[["train", "validation", "test"]].sum(axis=1)# sum train and validation columns
    df["train_validation_total"] = df["train"] + df["validation"]

    # compute costs 
    cost_per_1000_tokens_finetuning = 0.008
    cost_per_1000_tokens_inference_finetuned = 0.003
    cost_per_1000_tokens_inference_gpt35 = 0.0005
    cost_per_1000_tokens_inference_gpt4 = 0.01
    num_epochs = 3 

    df["training_cost"] = (df["train"] + df["validation"]) * cost_per_1000_tokens_finetuning /1000* num_epochs
    df["inference_cost"] = df["test"] * cost_per_1000_tokens_inference_finetuned / 1000
    df["total_cost"] = df["training_cost"] + df["inference_cost"]



    df["0shot_cost_gpt3.5"] = df["test"] * cost_per_1000_tokens_inference_gpt35  /1000 
    df["0shot_cost_gpt4"] = df["test"] * cost_per_1000_tokens_inference_gpt4 /1000

    print(df)

def format_and_save_data_for_gpt_finetuning(dataset: Dataset, split:str, save_dir:str) -> str: 
    
    dataset = dataset.map(lambda x : 
        {
            "sentence_only_messages": {
                "messages": 
                    [
                        {
                            "role": "system",
                            "content": classifier._form_system_prompt() 
                        },
                        {
                            "role": "user",
                            "content": classifier._form_target_sentence_only_prompt(x)
                        },
                        {
                            "role": "assistant",
                            "content": x["update_category"]
                        }
                    ],

            }, 
            "direct_context_messages": {
                "messages": 
                    [
                        {
                            "role": "system",
                            "content": classifier._form_system_prompt() 
                        },
                        {
                            "role": "user",
                            "content": classifier._form_target_with_direct_context_prompt(x)
                        },
                        {
                            "role": "assistant",
                            "content": x["update_category"]
                        }
                    ]
            },
            "full_article_messages": {
                "messages": 
                    [
                        {
                            "role": "system",
                            "content": classifier._form_system_prompt() 
                        },
                        {
                            "role": "user",
                            "content": classifier._form_target_with_full_article_prompt(x)
                        },
                        {
                            "role": "assistant",
                            "content": x["update_category"]
                        }
                    ]
            }
        }
    )
    
    
    fps = {}
    # get list of messages for each prompt type 
    for prompt_type in ["sentence_only", "direct_context", "full_article"]:
        messages_list = dataset[prompt_type+ "_messages"]
        
        fp = Path(save_dir) / f"{prompt_type}_{split}.jsonl"
        with fp.open("w") as f:
            for messages in messages_list: 
                f.write(json.dumps(messages) + "\n")
                
        fps[prompt_type] = fp

    return fps
    

def submit_finetuning_job(train_fp:str, suffix:str, model:str="gpt-3.5-turbo"): 
    
    client = OpenAI()
    
    # upload file
    upload_result = client.files.create(
        file=open(train_fp, "rb"),
        purpose="fine-tune"
    )
    
    logger.info(upload_result)
    
    # start finetuning
    finetune_result = client.fine_tuning.jobs.create(
        training_file=upload_result.id,
        model=model,
        suffix=suffix,
        hyperparameters={
            "n_epochs":1
        }
    )
    
    logger.info(finetune_result)
    
    client.fine_tuning.jobs.list(limit=10)


def main(): 

    parser = ArgumentParser()
    parser.add_argument("--prompt_type", type=str, default="sentence_only", help="prompt type to use for finetuning. One of 'sentence_only', 'direct_context', 'full_article'")
    parser.add_argument("--label_type", type=str, default="fact", help="Type of label to focus on. One of ['fact', 'strict_fact', 'all']. All looks at 'style' labels as well.")
    parser.add_argument("--format_data", action="store_true", help="format and save data for finetuning")
    parser.add_argument("--fp", type=str, help="file path to use for finetuning, should be a jsonl file with openai messages formatting")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="model to use for finetuning")
    args = parser.parse_args()
    parser.add_argument("--save_dir", type=str, default="openai_data", help="directory to save finetuning files")
    parser.add_argument("--version", type=int, default=3, help="version of the data to use for finetuning")
    parser.add_argument("--home_dir", type=str, default="/project/jonmay_231/hjcho/NewsEdits", help="home directory for the project")
    parser.add_argument("--train_size", type=int, default=16000, help="size of the training dataset")
    parser.add_argument("--validation_size", type=int, default=2000, help="size of the validation dataset")
    parser.add_argument("--seed", type=int, default=42, help="seed for shuffling the datasets")
    args = parser.parse_args()
    
    # calculate total costs for finetuning and inference
    # calc_finetuning_total_costs()
    # all_fps = {}


    classifier = OpenAIClassifier(model_name=args.model)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    version=args.version
    label_type = args.label_type
    train_size = args.train_size
    validation_size = args.validation_size
    seed = arg.seed
    HOME_DIR = args.home_dir
    
    assert label_type in ["fact", "strict_fact", "all"], f"label type: {label_type} not in ['fact', 'strict_fact', 'all']"

    # load data
    train_data = pd.read_csv(f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction-with-date-train{label_type}.csv")
    validation_data = pd.read_csv(f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction-with-date-dev{label_type}.csv")
    test_data = pd.read_csv(f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction-with-date-test{label_type}.csv")

    gold_test_data_fp = f"{HOME_DIR}/data/prediction_data/test_v2.jsonl"

    with open(gold_test_data_fp, "r") as f:
        gold_test_data = [json.loads(line) for line in f]
        
    gold_test_data = pd.DataFrame(gold_test_data)
    # create row with coarsegrained labels 

    # drop sentence with na labels
    finegrained_to_coarsegrained = {label:k for label in v for k, v in COARSEGRAINED_TO_FINEGRAINED.items()}
    gold_test_data = gold_test_data.dropna(subset=["source_sentence"])
    gold_test_data["update_category"] = gold_test_data["labels"].map(lambda x: "style" if x in COARSEGRAINED_TO_FINEGRAINED["style"] else "")

    gold_test_dataset = Dataset.from_pandas(test_data, split="test")

    train_dataset = Dataset.from_pandas(train_data, split="train")
    validation_dataset = Dataset.from_pandas(validation_data, split="validation")
    test_dataset = Dataset.from_pandas(test_data, split="test")

    # shuffle with seed to 42  and sample 16000
    train_dataset = train_dataset.shuffle(seed=seed).select(range(train_size))
    # do the same with 2000 for validation and test 
    validation_dataset = validation_dataset.shuffle(seed=seed).select(range(validation_size))
    test_dataset = test_dataset.shuffle(seed=seed).select(range(validation_size))

    # save these files as csv
    train_dataset.to_csv(f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction-with-date-train_{train_size}.csv")
    validation_dataset.to_csv(f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction-with-date-dev_{validation_size}.csv")
    test_dataset.to_csv(f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction-with-date-test_{validation_size}.csv")

    if args.format_data and args.fp is None: 
        if not Path(args.save_dir).exists(): 
            Path(args.save_dir).mkdir()
        
        validation_fps = format_and_save_data_for_gpt_finetuning(validation_dataset, "validation", save_dir=args.save_dir)
        test_fps = format_and_save_data_for_gpt_finetuning(test_dataset, "test", save_dir=args.save_dir)
        fps = format_and_save_data_for_gpt_finetuning(train_dataset, "train", save_dir=args.save_dir)
        
        fp = fps[args.prompt_type]
        logger.info(f"Files saved for finetuning: {fps}")
        return 
    
    else: 
        fp = args.fp
    
    # assert
    assert args.prompt_type in fp, f"prompt type: {args.prompt_type} not in file path: {fp}, check that the file path is correct or that the prompt type is correct."        
        
    logger.info(f"Submitting finetuning job for prompt type: {args.prompt_type} using file: {fp}")

    # submit for finetuning 
    submit_finetuning_job(fp, model=model, suffix=args.prompt_type)

    return 


if __name__ == "__main__":
    main()
