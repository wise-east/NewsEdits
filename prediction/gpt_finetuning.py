import pandas as pd
from transformers import AutoTokenizer 
from predict_label_type import OpenAIClassifier
from datasets import load_dataset, Dataset
from argparse import ArgumentParser
import numpy as np
from openai import OpenAI
import json
from loguru import logger


classifier = OpenAIClassifier(model_name="gpt-3.5-turbo")
tokenizer = AutoTokenizer.from_pretrained('gpt2')
version=3

# format data 
HOME_DIR = "/project/jonmay_231/hjcho/NewsEdits"

train_data = pd.read_csv(f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction-with-date-train.csv")
validation_data = pd.read_csv(f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction-with-date-dev.csv")
test_data = pd.read_csv(f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction-with-date-test.csv")

train_dataset = Dataset.from_pandas(train_data, split="train")
validation_dataset = Dataset.from_pandas(validation_data, split="validation")
test_dataset = Dataset.from_pandas(test_data, split="test")

# shuffle with seed to 42  and sample 16000
train_dataset = train_dataset.shuffle(seed=42).select(range(16000))
# do the same with 2000 for validation and test 
validation_dataset = validation_dataset.shuffle(seed=42).select(range(2000))
test_dataset = test_dataset.shuffle(seed=42).select(range(2000))


def calc_finetuning_total_costs(): 
    
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

def format_and_save_data_for_gpt_finetuning(dataset: Dataset, split:str) -> str: 
    
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
        
        fp = f"{prompt_type}_{split}.jsonl"
        with open(fp, "w") as f: 
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
    parser.add_argument("--format_data", action="store_true", help="format and save data for finetuning")
    parser.add_argument("--fp", type=str, help="file path to use for finetuning, should be a jsonl file with openai messages formatting")
    args = parser.parse_args()
    
    # calculate total costs for finetuning and inference
    # calc_finetuning_total_costs()
    # all_fps = {}
    # for dataset, split in [(train_dataset, "train"), (validation_dataset, "validation"), (test_dataset, "test")]:
    #     fp = format_and_save_data_for_gpt_finetuning(dataset, split)
    #     fps[split] = fp

    if args.format_data and args.fp is None: 

        fps = format_and_save_data_for_gpt_finetuning(train_dataset, "train")
        
        fp = fps[args.prompt_type]
        logger.info(f"Files saved for finetuning: {fps}")
        return 
    
    else: 
        fp = args.fp
    
    # assert
    assert args.prompt_type in fp, f"prompt type: {args.prompt_type} not in file path: {fp}, check that the file path is correct or that the prompt type is correct."        
        
    logger.info(f"Submitting finetuning job for prompt type: {args.prompt_type} using file: {fp}")

    # submit for finetuning 
    submit_finetuning_job(fp, suffix=args.prompt_type)

    return 


if __name__ == "__main__":
    main()
