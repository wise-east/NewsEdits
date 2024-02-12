# Author: Justin Cho 
# Email: jcho@isi.edu 
# Date: 2024-02-05
# Usage: python format_data.py
# Purpose: 

import pandas as pd
from loguru import logger
from datetime import datetime
import numpy as np 
from tqdm import tqdm 
from utils import test_key_agreement_before_merging, load_csv_as_df, format_date
import numpy as np

HOME_DIR = "/project/jonmay_231/hjcho/NewsEdits"

logger.info("Starting to format data...")
version = 3 

# load files to merge to get timestamps 
# data with article content
orig_df = load_csv_as_df(f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction.csv")
# metadata with timestamps
timestamp_df = load_csv_as_df(f"{HOME_DIR}/data/prediction_data/v2-metadata-incl-timestamp.csv")

logger.info("Loaded all data")

# check that version updates are always increments by 1 
# needed for formatting timestamp df, which only provides version for either version_x or version_y
orig_df['version_diff'] = orig_df['version_y'] - orig_df['version_x']
assert len(orig_df) == orig_df['version_diff'].value_counts().to_dict()[1]

# only keep the rows with version_type = version_x 
new_columns = {"version_x": [], "version_y": []}

# the two files have different names used for sources 
source_mappings = {
    "bbc": "bbc-2", 
    "nytimes": "nyt",
    "washpo": "wp"
}

# Map source names
timestamp_df["source"] = timestamp_df["source"].map(source_mappings).fillna(timestamp_df["source"])

# Calculate version_x and version_y based on version_type
timestamp_df["version_x"] = np.where(timestamp_df["version_type"] == "version_x", timestamp_df["version"], timestamp_df["version"] - 1)
timestamp_df["version_y"] = np.where(timestamp_df["version_type"] == "version_x", timestamp_df["version"] + 1, timestamp_df["version"])

# drop the version column
timestamp_df = timestamp_df.drop(columns=["version", "version_type"])

# optional as it has been tested for the files of intereset. can skip 
merge_keys = ['entry_id', 'source', 'version_x', 'version_y']
# test_key_agreement_before_merging(df1 = orig_df, df2 = timestamp_df, keys=merge_keys)

logger.info(f"# rows before merge: {len(orig_df)}")


# join the two dfs on such that the other columns are added to the rows that share the same ['entry_id', 'source', 'version_x]
timestamp_df_unique = timestamp_df.drop_duplicates(subset=merge_keys, keep='first')
df = orig_df.merge(timestamp_df_unique, on=merge_keys, how='left')
logger.info(f"Completed merge. # row after merge: {len(df)}")

# remove cases where the sentence is NaN (do this later)
df = df[~df["sentence"].isna()]

# drop duplicates in sentence column
df = df.drop_duplicates(subset=["sentence", "label", "entry_id", "source", "version_x", "version_y"])

# add article id column 
# create article_id column 
df = df.assign(
    article_id=df.index.to_series().groupby(
        [df.entry_id, df.source, df.version_x, df.version_y]
    ).transform('first')
)

# add direct context column that contains the prior, current, and the next sentence. only add them if their article ids are the same 
# Create a column for the prior sentence with a check to ensure article IDs match
df["prior_sentence"] = np.where(df["article_id"] == df["article_id"].shift(), df["sentence"].shift(), "")

# Create a column for the next sentence with a check to ensure article IDs match
df["next_sentence"] = np.where(df["article_id"] == df["article_id"].shift(-1), df["sentence"].shift(-1), "")

# Concatenate prior, current, and next sentences to form the direct_context
df["direct_context"] = df["prior_sentence"] + " " + df["sentence"] + " " + df["next_sentence"]

# Optionally, you might want to strip leading/trailing whitespace if prior_sentence or next_sentence are empty
df["direct_context"] = df["direct_context"].str.strip()

# drop prior_sentence and next_sentence columns
df = df.drop(columns=["prior_sentence", "next_sentence"])

# add full article column that contains the full article, excluding nan sentences and duplicates 
# do groupby first and then drop duplicates 
df["full_article"] = df["sentence"].groupby([df.entry_id, df.source, df.version_x, df.version_y]).transform(lambda x: " ".join(x.dropna().drop_duplicates()))

# make sure that there is no nan entry for the 'created' column
nan_created = df[df["created"].isna()]
assert len(nan_created) == 0, f"Number of nan entries for 'created' column: {len(nan_created)}"

# change nan labels to "none"
df = df.fillna({"label": "none"})

# get value counts of each label
label_counts = df["label"].value_counts()
# logger.info(label_counts)


# remove those that are <1000 
label_counts = label_counts[label_counts > 1000]
df = df[df["label"].isin(label_counts.index)]

# label types that introduce factual updates 
fact_update_labels = [
    'Update Background',
    'Delete Background',
    'Delete Quote',
    'Event Update',
    'Quote Update',
    'Quote Addition',
    'Event Addition',
    'Add Analysis',
    'Add Information (Other)',
    'Update Analysis',
    'Add Background',
    'Correction',
    'Delete Analysis',
    'Additional Sourcing',
    'Add Eye-witness account',
    '“Update Background',
    '“Add Analysis',
]

# remove "* background" labels to focus more on strict fact updates
fact_update_labels_v2 = [
    'Delete Quote',
    'Event Update',
    'Quote Update',
    'Quote Addition',
    'Event Addition',
    'Add Analysis',
    'Add Information (Other)',
    'Update Analysis',
    'Correction',
    'Delete Analysis',
    'Additional Sourcing',
    'Add Eye-witness account',
    '“Add Analysis',
    '(Additional Sourcing',
    '(Source-Document Update',
    '(Source-Document Addition',
    'Add Additional Sourcing',
    'Delete Event Reference',
    '“Add Eye-witness account' 
]

# label types that introduce stylistic updates 
style_update_labels = [
    'Tonal Edits',
    'Syntax Correction',
    'De-emphasize a Point',
    'Emphasize importance',
    'Style-Guide Edits',
    'Emphasize a Point',
    'De-emphasize importance',
    'Tonal Improvement',
    'Simplification',
]

# create a new column that contains update category 
df["update_category"] = np.where(df["label"].isin(fact_update_labels), "fact", np.where(df["label"].isin(style_update_labels), "style", "none"))

# get count of rows with fact_update_labels as the label
fact_update_df = df[df["label"].isin(fact_update_labels)]
# get count of rows with fact_update_labels_v2 as the label
fact_update_df_v2 = df[df["label"].isin(fact_update_labels_v2)]

# get count of rows with style_update_labels as the label
style_update_df = df[df["label"].isin(style_update_labels)]
none_update_df = df[df["label"] == "none"]

logger.info(f"\n# fact update rows: {len(fact_update_df)}\n# fact update v2 rows: {len(fact_update_df_v2)}\n# style update rows: {len(style_update_df)}\n# none update rows: {len(none_update_df)}")

all_df = pd.concat([fact_update_df, style_update_df, none_update_df])
all_df.created = all_df.created.apply(lambda x: format_date(x))

# sort by timestamp to get 8:1:1 split 
all_df = all_df.sort_values(by="created")

# get a list of article ids with orders preserved
article_ids = all_df["article_id"].drop_duplicates().tolist()

# sample 20,000 article ids in sorted order with split by article ids 8:1:1
train_articles = set(np.random.choice(article_ids[:int(len(article_ids) * 0.8)], 16_000, replace=False))
dev_articles = set(np.random.choice(article_ids[int(len(article_ids) * 0.8):int(len(article_ids) * 0.9)], 2_000, replace=False))
test_articles = set(np.random.choice(article_ids[int(len(article_ids) * 0.9):], 2_000, replace=False))

split_mapping = np.where(all_df["article_id"].isin(train_articles), "train", np.where(all_df["article_id"].isin(dev_articles), "dev", "test"))
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
train_df = train_df.groupby("update_category").apply(lambda x: x.sample(min_category_count)).reset_index(drop=True)

#verify that the number of rows for each update category is the same
update_category_counts_train = train_df["update_category"].value_counts()
logger.info(update_category_counts_train)

logger.info(f"\n# train rows: {len(train_df)}\n# dev rows: {len(dev_df)}\n# test rows: {len(test_df)}")

combined_label_distribution = pd.concat([train_df.label.value_counts(), dev_df.label.value_counts(), test_df.label.value_counts()], axis=1)
combined_label_distribution.columns = ["train", "dev", "test"]
logger.info(combined_label_distribution)

combined_update_category_distribution = pd.concat([train_df.update_category.value_counts(), dev_df.update_category.value_counts(), test_df.update_category.value_counts()], axis=1)
combined_update_category_distribution.columns = ["train", "dev", "test"]
logger.info(combined_update_category_distribution)

# # save as csv 
# save_fp = f"{HOME_DIR}/data/prediction_data/v2-silver-labeled-data-for-prediction-with-date-and-split.csv"
# all_df.to_csv(save_fp)
# logger.info(f"Saved data as csv file to: {save_fp}")

train_fp = f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction-with-date-train.csv"
dev_fp = f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction-with-date-dev.csv"
test_fp = f"{HOME_DIR}/data/prediction_data/v{version}-silver-labeled-data-for-prediction-with-date-test.csv"
train_df.to_csv(train_fp)
dev_df.to_csv(dev_fp)
test_df.to_csv(test_fp)