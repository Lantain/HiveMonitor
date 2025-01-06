import pandas as pd
import math
import numpy as np
from sklearn.utils import shuffle
import os
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None

def set_delta(i: int, delta_i: int, offset: int, df: pd.DataFrame, column: str, delta_column: str):
    if i > delta_i:
        try:
            df[f'{delta_column}{delta_i}']
        except:
            # print(f'CATCH {delta_column}{delta_i}')
            df[f'{delta_column}{delta_i}'] = 0.
        curr = df.iloc[i]
        prevDI = df.iloc[i - delta_i]
        perc_deltaDI = (curr[column] - prevDI[column]) / prevDI[column]
        df.at[i + offset, f'{delta_column}{delta_i}'] = perc_deltaDI
        

# Populate weight delta
def populate_delta(df: pd.DataFrame, offset: int = 0):
    df['weight_delta'] = df['weight_kg'].diff()
    df['weight_delta_percent'] = 0.

    for i in range(1, len(df)):
        if i == 0:
            continue
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        perc_delta = (curr['weight_kg'] - prev['weight_kg']) / prev['weight_kg']
        df.at[i + offset, 'weight_delta_percent'] = perc_delta
        # Calculate diffs
        set_delta(i, 3, offset, df, 'weight_kg', 'weight_perc_l')
        set_delta(i, 5, offset, df, 'weight_kg', 'weight_perc_l')
        set_delta(i, 10, offset, df, 'weight_kg', 'weight_perc_l')
        set_delta(i, 30, offset, df, 'weight_kg', 'weight_perc_l')
        
    return df

def populate_humidity_delta(df: pd.DataFrame, offset: int = 0):
    for i in range(1, len(df)):
        if i == 0:
            continue
        set_delta(i, 3, offset, df, 'h', 'h_perc_l')
        set_delta(i, 5, offset, df, 'h', 'h_perc_l')
        set_delta(i, 10, offset, df, 'h', 'h_perc_l')
        set_delta(i, 30, offset, df, 'h', 'h_perc_l')

def populate_temp_delta(df: pd.DataFrame, offset: int = 0):
    df['temp_io_diff'] = 0.
    # Get all temperature columns
    temps = [df['t_i_1'], df['t_i_2'], df['t_i_3'], df['t_i_4'], df['t_i_5']]
    
    # Calculate mean only for non-NaN values in each row
    df['temp_mid'] = pd.concat(temps, axis=1).apply(lambda x: x[~pd.isna(x)].mean(), axis=1)
    df['temp_io_diff_smart'] = np.where(
        pd.isna(df['temp_mid']) | pd.isna(df['t_o']), 
        np.nan,
        df['temp_mid'] - df['t_o']
    )
    df['t_delta_percent'] = 0.

    for i in range(1, len(df)):
        if i == 0:
            continue
        prev = df.iloc[i - 1]
        current = df.iloc[i]
        
        if prev['temp_mid'] == 0 or np.isnan(prev['temp_mid']):
            continue
        
        perc_delta = (current['temp_mid'] - prev['temp_mid']) / prev['temp_mid']
        df.at[i + offset, 't_delta_percent'] = perc_delta
        df.at[i + offset, 'temp_io_diff'] = current['temp_mid'] - current['t_o']

        set_delta(i, 3, offset, df, 'temp_mid', 't_perc_l')
        set_delta(i, 5, offset, df, 'temp_mid', 't_perc_l')
        set_delta(i, 10, offset, df, 'temp_mid', 't_perc_l')
        set_delta(i, 30, offset, df, 'temp_mid', 't_perc_l')
        # set_delta(i, 60, offset, df, 'temp_mid', 't_perc_l')
        # set_delta(i, 90, offset, df, 'temp_mid', 't_perc_l')
        # set_delta(i, 120, offset, df, 'temp_mid', 't_perc_l')


    # populate_normalized(df, 'temp_io_diff')
    # populate_normalized(df, 't_perc_l3')
    # populate_normalized(df, 't_perc_l5')
    # populate_normalized(df, 't_perc_l10')
    # populate_normalized(df, 't_perc_l30')
    # populate_normalized(df, 't_perc_l60')
    # populate_normalized(df, 't_perc_l90')

        # # diff = (c['t_i_1'] + c['t_i_2'] + c['t_i_3'] + c['t_i_4'] + c['t_i_5']) / 5 - c['t_o']
        # diff = (curr['t_i_1'] + curr['t_i_2']) / 2 - curr['t_o']
        # # print(f"{i}: T1: {c['t_i_1']:.1f}, T2: {c['t_i_2']:.1f}, TO: {c['t_o']:.1f} diff: {diff}")
        # df.at[i + offset, 'temp_io_diff'] = diff

def populate_swarming_column(df: pd.DataFrame, swarming_indexes):
    df['swarming'] = 0
    for index in swarming_indexes:
        df.at[index, 'swarming'] = 1

def populate_column_by_index(df: pd.DataFrame, column: str, indexes):
    df[column] = 0
    for index in indexes:
        df.at[index, column] = 1

def populate_normalized(df: pd.DataFrame, column: str):
    df[column+'_norm'] = 0.
    scaler = MinMaxScaler(feature_range=(-1., 1.))
    scaled = scaler.fit_transform(df[[column]])
    # scaled = scaled[np.isfinite(scaled).all(scaled.mean())]
    df[column+'_norm'] = scaled

def populate_scaled_weight(df: pd.DataFrame):
    df['weight_kg_scaled'] = df['weight_kg'] / 100

def to_split(df: pd.DataFrame, column: str, segment_size: int):
    i = 0
    margin = 3  #math.floor(segment_size / 8)
    if margin < 1:
        margin = 1
    print('Margin: ', margin)
    x = []
    y = []
    while i < len(df) - segment_size:
        segment = df.iloc[i:i+segment_size]
        is_swarming = 0
        for j in range(margin, segment_size-margin):
            # if segment.iloc[j]['swarming'] == 1 and j > margin and j < (segment_size - margin):
            if segment.iloc[j]['swarming'] == 1 and j > margin and j < (segment_size - margin):
                is_swarming = 1
                print(f"Swarming at {i} -> {j}")
                break
            
        x_entry = segment[column].to_numpy().reshape(segment_size, 1)
        if len(x_entry) == segment_size:
            x.append(x_entry)
            y.append([1 if is_swarming else 0])
            i += 1

    x, y = shuffle(x, y) # random_state=0)
    return x, y

def to_universal_split(df: pd.DataFrame, segment_size: int):
    i = 0
    margin = math.floor(segment_size / 20)
    if margin < 1:
        margin = 1
    print('Margin: ', margin)
    x = []
    y = []
    while i < len(df) - segment_size:
        segment = df.iloc[i:i+segment_size]
        
        is_queencell = 0
        is_feeding = 0
        is_honey = 0
        is_treatment = 0
        is_died = 0
        is_swarming = 0

        for j in range(margin, segment_size-margin):
            if segment.iloc[j]['swarming'] == 1 and j > margin and j < (segment_size - margin):
                is_swarming = 1
                # print(f"Swarming at {i} -> {j}")
                break
            if segment.iloc[j]['queencell'] == 1 and j > margin and j < (segment_size - margin):
                is_queencell = 1
                # print(f"Queen Cell at {i} -> {j}")
                break
            if segment.iloc[j]['feeding'] == 1 and j > margin and j < (segment_size - margin):
                is_feeding = 1
                # print(f"Feeding at {i} -> {j}")
                break
            if segment.iloc[j]['honey'] == 1 and j > margin and j < (segment_size - margin):
                is_honey = 1
                # print(f"Honey at {i} -> {j}")
                break
            if segment.iloc[j]['treatment'] == 1 and j > margin and j < (segment_size - margin):
                is_treatment = 1
                # print(f"Treatment at {i} -> {j}")
                break
            if segment.iloc[j]['died'] == 1 and j > margin and j < (segment_size - margin):
                is_died = 1
                # print(f"Died at {i} -> {j}")
                break
        
        segment['temp_mid'] = segment['temp_mid'] / 60
        segment['h'] = segment['h'] / 100
        segment['weight_kg'] = segment['weight_kg'] / 80
        
        x_entry = segment[[
            # 'temp_mid',
            # 'h', 
            # 'weight_kg', 
            'weight_perc_l10', 
            # 't_perc_l10',
            # 'h_perc_l10',
        ]].to_numpy() #.reshape(segment_size, 1)

        if len(x_entry) == segment_size:
            x.append(x_entry)
            
            y.append([(
                1. if is_queencell else 0.,
                1. if is_swarming else 0.,
                1. if is_honey else 0.,
                1. if is_feeding else 0.,
                1. if is_treatment else 0.,
                1. if is_died else 0.
            )])
            # y.append([{
            #     'queencell_output': 1 if is_queencell else 0,
            #     'feeding_output': 1 if is_feeding else 0,
            #     'honey_output': 1 if is_honey else 0,
            #     'treatment_output': 1 if is_treatment else 0,
            #     'died_output': 1 if is_died else 0,
            #     'swarming_output': 1 if is_swarming else 0,
            # }])
            i += margin
    x, y = shuffle(x, y) #random_state=0)
    
    return x, y

def dir_to_split(dir: str, column: str, segment_size: int):
    files = os.scandir(dir)
    x = list()
    y = list()
    for file in files:
        print('Processing: ', file.path)
        df = pd.read_csv(file.path)
        if column == 'temp_mid' or column == 'temp_io_diff_smart':
            df[column] = df[column] / 60
        if column == 'h':
            df[column] = df[column] / 100
        if column == 'weight_kg':
            df[column] = df[column] / 80
        xs, ys = to_split(df, column, segment_size)
        x = x + xs
        y = y + ys

    return np.array(x), np.array(y)

def universal_dir_to_split(dir: str, segment_size: int):
    files = os.scandir(dir)
    x = list()
    y = list()
    for file in files:
        print('Processing: ', file.path)
        df = pd.read_csv(file.path)
        xs, ys = to_universal_split(df, segment_size)
        x = x + xs
        y = y + ys

    return np.array(x), np.array(y)
    
# def smooth_col(col, threshold=0.3):
#     c = col.copy()
#     c[abs(c.diff()) > threshold] = np.nan
#     c.interpolate(method='linear', inplace=True)
#     return c

def smooth_col(col, threshold=0.3):
    c = col.copy()
    c[abs(c.diff()) > threshold] = np.nan
    # Convert to numeric type before interpolating
    c = pd.to_numeric(c, errors='coerce')
    c.interpolate(method='linear', inplace=True)
    return c

def balanceXandY(x, y):
    # Assume x and y are your input arrays
    # y has 22% of elements as 1 and 78% as 0

    # Separate indices where Y is 1 and Y is 0
    y_1_indices = np.where(y == 1)[0]
    y_0_indices = np.where(y == 0)[0]

    # Find the smaller subset size to balance the data
    n_samples = min(len(y_1_indices), len(y_0_indices))

    # Randomly sample indices from each subset
    y_1_sample_indices = np.random.choice(y_1_indices, n_samples, replace=False)
    y_0_sample_indices = np.random.choice(y_0_indices, n_samples, replace=False)

    # Combine the indices and shuffle them
    balanced_indices = np.concatenate([y_1_sample_indices, y_0_sample_indices])
    np.random.shuffle(balanced_indices)

    # Create balanced X and Y arrays
    X_balanced = x[balanced_indices]
    Y_balanced = y[balanced_indices]
    return X_balanced, Y_balanced

def validate_part_events(df: pd.DataFrame, fname: str):
    has_swarming = len(df[df['swarming'] == 1].index) != 0
    has_queencell = len(df[df['queencell'] == 1].index) != 0
    has_feeding = len(df[df['feeding'] == 1].index) != 0
    has_honey = len(df[df['honey'] == 1].index) != 0
    has_treatment = len(df[df['treatment'] == 1].index) != 0
    has_died = len(df[df['died'] == 1].index) != 0


    if has_swarming and (has_died or has_treatment or has_honey or has_feeding or has_queencell):
        print(f"Skipping {fname}: Swarming with other events")
        return False

    if has_queencell and (has_died or has_treatment or has_honey or has_feeding or has_swarming):
        print(f"Skipping {fname}: Queencell with other events")
        return False

    if has_feeding and (has_died or has_treatment or has_honey or has_queencell or has_swarming):
        print(f"Skipping {fname}: Feeding with other events")
        return False

    if has_honey and (has_died or has_treatment or has_feeding or has_queencell or has_swarming):
        print(f"Skipping {fname}: Honey with other events")
        return False

    if has_treatment and (has_died or has_honey or has_feeding or has_queencell or has_swarming):
        print(f"Skipping {fname}: Treatment with other events")
        return False

    if has_died and (has_treatment or has_honey or has_feeding or has_queencell or has_swarming):
        print(f"Skipping {fname}: Died with other events")
        return False
    return True

def event_shorts_for_part(df: pd.DataFrame):
    has_swarming = len(df[df['swarming'] == 1].index) != 0
    has_queencell = len(df[df['queencell'] == 1].index) != 0
    has_feeding = len(df[df['feeding'] == 1].index) != 0
    has_honey = len(df[df['honey'] == 1].index) != 0
    has_treatment = len(df[df['treatment'] == 1].index) != 0
    has_died = len(df[df['died'] == 1].index) != 0

    return f"{'s' if has_swarming else ''}{'q' if has_queencell else ''}{'f' if has_feeding else ''}{'h' if has_honey else ''}{'t' if has_treatment else ''}{'d' if has_died else ''}"

def read_dataset_file(file: str):
    df = pd.read_csv(file, dtype={
        't_i_1': float,
        't_i_2': float,
        't_i_3': float,
        't_i_4': float,
        't_o': float,
        'weight_kg': float,
        "weight_delta": float,
        'numeric.time': float,
        'h': float,
        't': float,
        'p': float,
    }, low_memory=False, parse_dates=['time'], index_col='time', date_format='%Y-%m-%d %H:%M:%S')
    return df

def fill_with_historical_pattern(df, columns, hours_ago=24):
    # Convert index to datetime if it's not already
    df.index = pd.to_datetime(df.index)
    
    # Calculate the time shift
    time_shift = pd.Timedelta(hours=hours_ago)
    
    for col in columns:
        # Find rows with NaN values
        nan_mask = df[col].isna()
        
        if not nan_mask.any():
            continue
            
        # For each NaN value
        for idx in df[nan_mask].index:
            # Get the same time of day from previous period
            historical_idx = idx - time_shift
            
            # If we have data from that time
            if historical_idx in df.index and not pd.isna(df.loc[historical_idx, col]):
                df.loc[idx, col] = df.loc[historical_idx, col]

    return df