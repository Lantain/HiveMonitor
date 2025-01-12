import json
TIMESTAMPS_FILE = 'timestamps-v4.json'

def get_all_timestamps():
    with open(TIMESTAMPS_FILE, 'r') as file:
        data = json.load(file)
    return data

def get_filename_from_timestamp(timestamp):
    return f"{timestamp['year']}_{timestamp['hive_number']}__{timestamp['from_month']}-{timestamp['from_day']}={timestamp['to_month']}-{timestamp['to_day']}.csv"

def get_stamps_with_event(event):
    stamps = get_all_timestamps()
    return [stamp for stamp in stamps if len(stamp[event]) > 0]

def sort_timestamps(stamps):
    return sorted(stamps, key=lambda x: int(x['year']) * 100 + int(x['hive_number']))

def save_timestamps(stamps):
    with open(TIMESTAMPS_FILE, 'w') as file:
        json.dump(stamps, file)
