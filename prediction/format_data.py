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

logger.info("Starting to format data...")

# load files to merge to get timestamps 
# data with article content
df = load_csv_as_df("/Users/jcho/projects/NewsEdits/data/prediction_data/v2-silver-labeled-data-for-prediction.csv")
# metadata with timestamps
timestamp_df = load_csv_as_df("/Users/jcho/projects/NewsEdits/data/prediction_data/v2-metadata-incl-timestamp.csv")

logger.info("Loaded all data")

# check that version updates are always increments by 1 
# needed for formatting timestamp df, which only provides version for either version_x or version_y
df['version_diff'] = df['version_y'] - df['version_x']
assert len(df) == df['version_diff'].value_counts().to_dict()[1]

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
test_key_agreement_before_merging(df1 = df, df2 = timestamp_df, keys=merge_keys)

# join the two dfs on such that the other columns are added to the rows that share the same ['entry_id', 'source', 'version_x]
df = df.merge(timestamp_df, on=merge_keys, how='left')
logger.info("Completed merge.")

# make sure that there is no nan entry for the 'created' column
nan_created = df[df["created"].isna()]
assert len(nan_created) == 0, f"Number of nan entries for 'created' column: {len(nan_created)}"

# remove cases where the sentence is NaN (do this later)
# df = df[~df["sentence"].isna()]

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
df["update_category"] = np.where(df["label"].isin(fact_update_labels), "fact", np.where(df["label"].isin(style_update_labels), "style", ""))

# get count of rows with fact_update_labels as the label
fact_update_df = df[df["label"].isin(fact_update_labels)]
# get count of rows with fact_update_labels_v2 as the label
fact_update_df_v2 = df[df["label"].isin(fact_update_labels_v2)]

# get count of rows with style_update_labels as the label
style_update_df = df[df["label"].isin(style_update_labels)]

logger.info(f"\n# fact update rows: {len(fact_update_df)}\n# fact update v2 rows: {len(fact_update_df_v2)}\n# style update rows: {len(style_update_df)}")


all_df = pd.concat([fact_update_df, style_update_df])
all_df.created = all_df.created.apply(lambda x: format_date(x))

# create article_id column 
all_df = all_df.assign(
    article_id=all_df.index.to_series().groupby(
        [all_df.entry_id, all_df.source, all_df.version_x, all_df.version_y]
    ).transform('first')
)

# sort by timestamp to get 8:1:1 split 
all_df = all_df.sort_values(by="created")

# get a list of article ids with orders preserved
article_ids = all_df["article_id"].drop_duplicates().tolist()
        
# spit by article ids 8:1:1 
train_articles = set(article_ids[:int(len(article_ids) * 0.8)])
dev_articles = set(article_ids[int(len(article_ids) * 0.8):int(len(article_ids) * 0.9)])
test_articles = set(article_ids[int(len(article_ids) * 0.9):])

train_df = all_df[all_df["article_id"].isin(train_articles)]
dev_df = all_df[all_df["article_id"].isin(dev_articles)]
test_df = all_df[all_df["article_id"].isin(test_articles)]

split_mapping = np.where(all_df["article_id"].isin(train_articles), "train", np.where(all_df["article_id"].isin(dev_articles), "dev", "test"))
all_df["split"] = split_mapping

logger.info(f"\n# train rows: {len(train_df)}\n# dev rows: {len(dev_df)}\n# test rows: {len(test_df)}")

combined_label_distribution = pd.concat([train_df.label.value_counts(), dev_df.label.value_counts(), test_df.label.value_counts()], axis=1)
combined_label_distribution.columns = ["train", "dev", "test"]
logger.info(combined_label_distribution)

# save as csv 
save_fp = "/Users/jcho/projects/NewsEdits/data/prediction_data/v2-silver-labeled-data-for-prediction-with-date-and-split.csv"
all_df.to_csv(save_fp)

logger.info(f"Saved data as csv file to: {save_fp}")
