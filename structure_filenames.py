import pandas as pd
import math
import events
import os
import json
import dataset

MINUTES = 15
SRC_DIR = 'dataset/bob_all_processed_mins'
MINUTES_DIR = f'dataset/transformed_minutes/interval_{str(MINUTES)}m'
TIMESTAMP_DIR = f'dataset/transformed_timestamps/interval_{str(MINUTES)}m'
POPULATED_DIR = f'dataset/transformed_populated/interval_{str(MINUTES)}m'
FILLED_DIR = f'dataset/transformed_filled/interval_{str(MINUTES)}m'
TIMESTAMPS = json.load(open('timestamps.json'))

files = os.scandir(TIMESTAMP_DIR)

for stamp in TIMESTAMPS:
    name = f"{stamp['year']}_{stamp['hive_number']}.csv"
    print(f"Processing {name}")
    df = dataset.read_dataset_file(os.path.join(MINUTES_DIR, name))
    df = df[pd.Timestamp(stamp['date_from']):pd.Timestamp(stamp['date_to'])]
    events_indexes: list[list[pd.Timestamp]] = events.get_parsed_event_indexes(df)
    # Convert Timestamp objects to strings before storing in JSON
    stamp['swarming'] = [i.strftime('%Y-%m-%d %H:%M') for i in events_indexes[0]]
    stamp['queencell'] = [i.strftime('%Y-%m-%d %H:%M') for i in events_indexes[1]] 
    stamp['feeding'] = [i.strftime('%Y-%m-%d %H:%M') for i in events_indexes[2]]
    stamp['honey'] = [i.strftime('%Y-%m-%d %H:%M') for i in events_indexes[3]]
    stamp['treatment'] = [i.strftime('%Y-%m-%d %H:%M') for i in events_indexes[4]]
    stamp['died'] = [i.strftime('%Y-%m-%d %H:%M') for i in events_indexes[5]]
    
json.dump(TIMESTAMPS, open('timestamps-v3.json', 'w+'))