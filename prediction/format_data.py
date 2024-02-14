# Author: Justin Cho
# Email: jcho@isi.edu
# Date: 2024-02-05
# Usage: python format_data.py
# Purpose: merge timestamp data with the silver label data and format it to contain the right columns for the prediction task (predict_label_type.py) and gpt_finetuning.py) and finetuning

import pandas as pd
from loguru import logger
from datetime import datetime
import numpy as np
from tqdm import tqdm
from utils import (
    test_key_agreement_before_merging,
    load_csv_as_df,
    format_date,
    create_finegrained_to_coarsegrained_map,
    COARSEGRAINED_TO_FINEGRAINED,
)
from argparse import ArgumentParser
import numpy as np
import json
import spacy
from pathlib import Path


def add_context_columns(df):

    # add direct context column that contains the prior, current, and the next sentence. only add them if their article ids are the same
    # Create a column for the prior sentence with a check to ensure article IDs match
    df["prior_sentence"] = np.where(
        df["article_id"] == df["article_id"].shift(), df["sentence"].shift(), ""
    )

    # Create a column for the next sentence with a check to ensure article IDs match
    df["next_sentence"] = np.where(
        df["article_id"] == df["article_id"].shift(-1), df["sentence"].shift(-1), ""
    )

    # Concatenate prior, current, and next sentences to form the direct_context
    df["direct_context"] = (
        df["prior_sentence"] + " " + df["sentence"] + " " + df["next_sentence"]
    )

    # Optionally, you might want to strip leading/trailing whitespace if prior_sentence or next_sentence are empty
    df["direct_context"] = df["direct_context"].str.strip()

    # drop prior_sentence and next_sentence columns
    df = df.drop(columns=["prior_sentence", "next_sentence"])

    # add full article column that contains the full article, excluding nan sentences and duplicates
    # do groupby first and then drop duplicates
    df["full_article"] = (
        df["sentence"]
        .groupby([df.entry_id, df.source, df.version_x, df.version_y])
        .transform(lambda x: " ".join(x.dropna().drop_duplicates()))
    )

    return df


def format_gold_label_data(gold_test_data_fp, finegrained_to_coarsegrained, label_type):
    logger.info("Starting to format gold label data...")

    # process the gold test data

    with open(gold_test_data_fp, "r") as f:
        gold_test_data = [json.loads(line) for line in f]

    gold_test_data = pd.DataFrame(gold_test_data)
    # drop na sentences
    gold_test_data = gold_test_data[~gold_test_data["source_sentence"].isna()]
    # if row's label is an empty list, set it to "none"
    gold_test_data["label"] = gold_test_data["label"].applymap(
        lambda x: x if x else ["none"]
    )

    for idx, row in gold_test_data.iterrows():
        finegrained_labels = row["label"]
        coarse_grained_label = set(
            [
                finegrained_to_coarsegrained.get(l.lower(), "none")
                for l in finegrained_labels
            ]
        )
        if "none" in coarse_grained_label and len(coarse_grained_label) > 1:
            coarse_grained_label = coarse_grained_label.difference({"none"})

        if len(coarse_grained_label) > 1:
            # if one of them is not none, keep that one
            logger.info(
                f"Coarse grained label: {coarse_grained_label} for fine grained labels: {finegrained_labels}"
            )
            coarse_grained_label = list(coarse_grained_label)
            breakpoint()

        elif len(coarse_grained_label) == 1:
            coarse_grained_label = list(coarse_grained_label)[0]

        if not isinstance(coarse_grained_label, str):
            breakpoint()

        gold_test_data.loc[idx, "update_category"] = coarse_grained_label

        sentence = row["source_sentence"]
        full_article = row["source_doc"]

        if sentence not in full_article:
            logger.info(
                f"Source sentence: {row['source_sentence']} not in full article: {row['source_doc']}"
            )

        # get the prior sentence
        prior_text = full_article[: full_article.find(sentence)].strip()

        # use spacy to sentence tokenize prior sentence and get the last item
        nlp = spacy.load("en_core_web_sm")

        prior_sentence = (
            list(nlp(prior_text).sents)[-1].text if prior_text != "" else ""
        )

        # get the next sentence
        next_text = full_article[full_article.find(sentence) + len(sentence) :].strip()

        next_sentence = list(nlp(next_text).sents)[0].text if next_text != "" else ""

        direct_context = f"{prior_sentence} {sentence} {next_sentence}".strip()
        gold_test_data.loc[idx, "direct_context"] = direct_context

    # rename source sentence column to sentence and source_doc to full_article
    gold_test_data = gold_test_data.rename(
        columns={"source_sentence": "sentence", "source_doc": "full_article"}
    )

    # get label distribution
    label_distribution = gold_test_data["label"].value_counts()

    # update category distribution
    update_category_distribution = gold_test_data["update_category"].value_counts()

    print(label_distribution)
    print(update_category_distribution)

    # save as csv
    save_fp = Path(gold_test_data_fp).with_name(f"gold_test_v2_{label_type}.csv")

    gold_test_data.to_csv(save_fp)
    return gold_test_data


def main():

    parser = ArgumentParser()
    parser.add_argument(
        "--version", type=int, default=3, help="Version of the data to format"
    )
    parser.add_argument(
        "--label_type",
        type=str,
        default="fact",
        help="Type of label to focus on. One of ['fact', 'strict_fact', 'all']. All looks at 'style' labels as well.",
    )
    parser.add_argument(
        "--home_dir",
        type=str,
        default="/project/jonmay_231/hjcho/NewsEdits",
        help="Home directory for the project",
    )
    parser.add_argument(
        "--num_train_articles",
        type=int,
        default=80_000,
        help="Number of articles to sample for the training set",
    )
    parser.add_argument(
        "--num_dev_articles",
        type=int,
        default=4_000,
        help="Number of articles to sample for the dev set",
    )
    args = parser.parse_args()

    HOME_DIR = args.home_dir
    version = args.version
    label_type = args.label_type

    finegrained_to_coarsegrained = create_finegrained_to_coarsegrained_map(
        COARSEGRAINED_TO_FINEGRAINED, label_type
    )

    gold_test_data_fp = f"{HOME_DIR}/data/prediction_data/test_v2.jsonl"
    # format_gold_label_data(gold_test_data_fp, finegrained_to_coarsegrained, label_type)

    logger.info("Starting to format silver label data...")

    # load files to merge to get timestamps
    # data with article content
    orig_df = load_csv_as_df(
        f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction.csv"
    )
    # metadata with timestamps
    timestamp_df = load_csv_as_df(
        f"{HOME_DIR}/data/prediction_data/v2-metadata-incl-timestamp.csv"
    )

    logger.info("Loaded all data")

    # check that version updates are always increments by 1
    # needed for formatting timestamp df, which only provides version for either version_x or version_y
    orig_df["version_diff"] = orig_df["version_y"] - orig_df["version_x"]
    assert len(orig_df) == orig_df["version_diff"].value_counts().to_dict()[1]

    # only keep the rows with version_type = version_x
    new_columns = {"version_x": [], "version_y": []}

    # the two files have different names used for sources
    source_mappings = {"bbc": "bbc-2", "nytimes": "nyt", "washpo": "wp"}

    # Map source names
    timestamp_df["source"] = (
        timestamp_df["source"].map(source_mappings).fillna(timestamp_df["source"])
    )

    # Calculate version_x and version_y based on version_type
    timestamp_df["version_x"] = np.where(
        timestamp_df["version_type"] == "version_x",
        timestamp_df["version"],
        timestamp_df["version"] - 1,
    )
    timestamp_df["version_y"] = np.where(
        timestamp_df["version_type"] == "version_x",
        timestamp_df["version"] + 1,
        timestamp_df["version"],
    )

    # drop the version column
    timestamp_df = timestamp_df.drop(columns=["version", "version_type"])

    # optional as it has been tested for the files of intereset. can skip
    merge_keys = ["entry_id", "source", "version_x", "version_y"]
    # test_key_agreement_before_merging(df1 = orig_df, df2 = timestamp_df, keys=merge_keys)

    logger.info(f"# rows before merge: {len(orig_df)}")

    # join the two dfs on such that the other columns are added to the rows that share the same ['entry_id', 'source', 'version_x]
    timestamp_df_unique = timestamp_df.drop_duplicates(subset=merge_keys, keep="first")
    df = orig_df.merge(timestamp_df_unique, on=merge_keys, how="left")
    logger.info(f"Completed merge. # row after merge: {len(df)}")
    df = df.fillna({"label": "none"})
    df["update_category"] = df["label"].apply(
        lambda x: finegrained_to_coarsegrained.get(x.lower(), "none")
    )
    print(df.value_counts("update_category"))

    # remove cases where the sentence is NaN (do this later)
    df = df[~df["sentence"].isna()]
    # change nan labels to "none"

    # drop duplicates in sentence column
    df = df.drop_duplicates(
        subset=["sentence", "label", "entry_id", "source", "version_x", "version_y"]
    )

    print(df.value_counts("update_category"))

    # add article id column
    # create article_id column
    df = df.assign(
        article_id=df.index.to_series()
        .groupby([df.entry_id, df.source, df.version_x, df.version_y])
        .transform("first")
    )

    # make sure that there is no nan entry for the 'created' column
    nan_created = df[df["created"].isna()]
    assert (
        len(nan_created) == 0
    ), f"Number of nan entries for 'created' column: {len(nan_created)}"

    # change nan labels to "none"
    df = df.fillna({"label": "none"})

    # get value counts of each label
    label_counts = df["label"].value_counts()
    # logger.info(label_counts)

    # remove those that are <1000
    # label_counts = label_counts[label_counts > 1000]
    df = df[df["label"].isin(label_counts.index)]

    df["update_category"] = df["label"].apply(
        lambda x: finegrained_to_coarsegrained.get(x.lower(), "none")
    )

    # get count of rows with fact_update_labels as the label
    fact_update_df = df[df["update_category"] == "fact"]
    # get count of rows with style_update_labels as the label
    style_update_df = df[df["update_category"] == "style"]
    none_update_df = df[df["update_category"] == "none"]

    logger.info(
        f"\n# fact update rows: {len(fact_update_df)}\n\n# style update rows: {len(style_update_df)}\n# none update rows: {len(none_update_df)}"
    )

    all_df = pd.concat([fact_update_df, style_update_df, none_update_df])
    all_df.created = all_df.created.apply(lambda x: format_date(x))
    # sort by timestamp
    all_df = all_df.sort_values(by="created")

    all_df = add_context_columns(all_df)

    # get a list of article ids with orders preserved
    article_ids = all_df["article_id"].drop_duplicates().tolist()

    # sample 20,000 article ids in sorted order with split by article ids 8:1:1
    train_articles = set(
        np.random.choice(
            article_ids[: int(len(article_ids) * 0.8)],
            args.num_train_articles,
            replace=False,
        )
    )
    dev_articles = set(
        np.random.choice(
            article_ids[int(len(article_ids) * 0.8) : int(len(article_ids) * 0.9)],
            args.num_dev_articles,
            replace=False,
        )
    )
    test_articles = set(
        np.random.choice(
            article_ids[int(len(article_ids) * 0.9) :],
            args.num_dev_articles,
            replace=False,
        )
    )

    split_mapping = np.where(
        all_df["article_id"].isin(train_articles),
        "train",
        np.where(all_df["article_id"].isin(dev_articles), "dev", "test"),
    )
    all_df["split"] = split_mapping

    train_df = all_df[all_df["article_id"].isin(train_articles)]
    dev_df = all_df[all_df["article_id"].isin(dev_articles)]
    test_df = all_df[all_df["article_id"].isin(test_articles)]

    # under sample the majority class to balance the classes for the training set
    # get the number of rows for each label
    update_category_counts_train = train_df["update_category"].value_counts()
    min_category = update_category_counts_train.idxmin()
    min_category_count = update_category_counts_train[min_category]

    # match the number of rows for each update category to the minimum number of rows for the update category
    train_df = (
        train_df.groupby("update_category")
        .apply(lambda x: x.sample(min_category_count))
        .reset_index(drop=True)
    )

    # verify that the number of rows for each update category is the same
    update_category_counts_train = train_df["update_category"].value_counts()
    logger.info(update_category_counts_train)

    logger.info(
        f"\n# train rows: {len(train_df)}\n# dev rows: {len(dev_df)}\n# test rows: {len(test_df)}"
    )

    combined_label_distribution = pd.concat(
        [
            train_df.label.value_counts(),
            dev_df.label.value_counts(),
            test_df.label.value_counts(),
        ],
        axis=1,
    )
    combined_label_distribution.columns = ["train", "dev", "test"]
    logger.info(combined_label_distribution)

    combined_update_category_distribution = pd.concat(
        [
            train_df.update_category.value_counts(),
            dev_df.update_category.value_counts(),
            test_df.update_category.value_counts(),
        ],
        axis=1,
    )
    combined_update_category_distribution.columns = ["train", "dev", "test"]
    logger.info(combined_update_category_distribution)

    # # save as csv
    # save_fp = f"{HOME_DIR}/data/prediction_data/v2-silver-labeled-data-for-prediction-with-date-and-split.csv"
    # all_df.to_csv(save_fp)
    # logger.info(f"Saved data as csv file to: {save_fp}")

    train_fp = f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction-with-date-train_{label_type}.csv"
    dev_fp = f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction-with-date-dev_{label_type}.csv"
    test_fp = f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction-with-date-test_{label_type}.csv"
    train_df.to_csv(train_fp)
    dev_df.to_csv(dev_fp)
    test_df.to_csv(test_fp)


if __name__ == "__main__":
    main()
