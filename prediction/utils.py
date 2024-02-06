import pandas as pd
from datetime import datetime 
from loguru import logger

# change timestamp to datetime object (e.g. 2021-01-07 05:28:00+00:00)
def format_date(datestr:str)->datetime: 
    # remove anything that comes after . or + 
    datestr = datestr.split(".")[0]
    datestr = datestr.split("+")[0]
    return datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S")

def load_csv_as_df(fp:str) -> pd.DataFrame: 
    return pd.read_csv(fp)

def test_key_agreement_before_merging(df1: pd.DataFrame, df2: pd.DataFrame, keys: list[str] = ['source', 'entry_id', 'version_x', 'version_y']) -> None: 
    # check that the resulting number of groupings are the same 
    grouped_df1 = len(df1.groupby(keys))
    grouped_df2 = len(df2.groupby(keys)) 
    logger.info(f"Number of article versions: {grouped_df1}\nNumber of article versions with timestamp: {grouped_df2}")

    assert grouped_df1 == grouped_df2, f"Number of groupings for df1 ({grouped_df1}) and number of groupings for df2 ({grouped_df2}) are not the same" 

    # check whether the set of values for keys are the same for both dfs
    df1_values = df1[keys].values
    df1_values = set([tuple(row) for row in df1_values])
    df2_values = df2[keys].values
    df2_values = set([tuple(row) for row in df2_values])
    are_sets_equal = df1_values == df2_values
    assert are_sets_equal, f"Sets of values for {keys} are not the same for both dfs"
