import math
import pandas as pd

def get_event_indexes(df: pd.DataFrame, event: str):
    indexes = []
    for i in range(1, len(df)):
        if i == 0:
            continue

        prev_diff = df.iloc[i-1][event]
        if math.isnan(prev_diff):
            prev_diff = None
        curr_diff = df.iloc[i][event]
        if math.isnan(curr_diff):
            curr_diff = None
        if prev_diff or curr_diff:
            if prev_diff is None and curr_diff is not None:
                indexes.append(i)
            elif curr_diff is None and prev_diff is not None:
                indexes.append(i)
            elif curr_diff > prev_diff:
                indexes.append(i)
    return indexes

def populate_swarming_column(df: pd.DataFrame, swarming_indexes):
    df['swarming'] = 0
    for index in swarming_indexes:
        df.at[index, 'swarming'] = 1