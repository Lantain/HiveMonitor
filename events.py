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
    
def get_raw_event_indexes(df: pd.DataFrame, event: str):
    event_col = f'{event}.next.dif'
    
    # Skip if column doesn't exist
    if event_col not in df.columns:
        return False
        
    # Create event column if it doesn't exist
    if event not in df.columns:
        df.loc[:, event] = 0
    
    # Get indexes where:
    # 1. Previous value exists (not NA)
    # 2. AND (current value is NA OR current value > previous value)
    event_mask = (
        df[event_col].shift(1).notna() & 
        (
            df[event_col].isna() |
            (df[event_col] > df[event_col].shift(1))
        ) &
        df['X.1'].notna()
    )
    
    # Get the indexes where events occur
    event_indexes = df[event_mask].index.tolist()
    return event_indexes

def populate_event_column(df: pd.DataFrame, event: str):
    event_indexes = get_raw_event_indexes(df, event)
    
    # Set values if we found any events
    if len(event_indexes) > 0:
        df.loc[event_indexes, event] = 1
        return True
        
    return False

def get_events_indexes(df: pd.DataFrame):
    events = ['swarming', 'queencell', 'feeding', 'honey', 'treatment', 'died']
    indexes = []
    for event in events:
       event_indexes = get_raw_event_indexes(df, event)
       indexes.append(event_indexes)
    return indexes

def get_parsed_event_indexes(df: pd.DataFrame):
    events = ['swarming', 'queencell', 'feeding', 'honey', 'treatment', 'died']
    indexes = []
    for event in events:
        event_indexes = df[df[event] == 1].index.tolist()
        indexes.append(event_indexes)
    return indexes