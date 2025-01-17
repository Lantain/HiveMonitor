import json
import pandas as pd
import dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import random

TIMESTAMPS_FILE = 'timestamps-v4.json'
STEP = 1
FOCUS_MARGIN=6


def get_all_timestamps(file=TIMESTAMPS_FILE):
    with open(TIMESTAMPS_FILE, 'r') as file:
        data = json.load(file)
    return data

def get_filename_from_timestamp(timestamp):
    date_from: pd.Timestamp = pd.to_datetime(timestamp['date_from'])
    date_to:pd.Timestamp = pd.to_datetime(timestamp['date_to'])
    
    return f"{timestamp['year']}_{timestamp['hive_number']}__{date_from.month}-{date_from.day}={date_to.month}-{date_to.day}.csv"

def sort_timestamps(stamps):
    return sorted(stamps, key=lambda x: int(x['year']) * 100 + int(x['hive_number']))

def save_timestamps(stamps):
    with open(TIMESTAMPS_FILE, 'w') as file:
        json.dump(stamps, file)

def segments_to_features_and_labels(segments):
    features = []
    labels = []
    for segment in segments:
        features.append(segment.get_data().fillna(0).to_numpy())
        labels.append(np.array(segment.get_label_tulp()))
    return np.array(features), np.array(labels)

def is_date_in_range(date: str, left: pd.Timestamp, right: pd.Timestamp):
    date = pd.Timestamp(date)
    return date >= left and date <= right


class BeeSegment:
    def __init__(self, timestamp, segment: pd.DataFrame, focus_margin: int = 2, features: list[str] = [], events: list[str] = []):
        self.timestamp: dict = timestamp
        self.segment = segment
        self.events = events
        # Use focused for identification to prevent useless events on borders
        self.segment_focus = segment.iloc[focus_margin:-focus_margin]
        self.left_segment_date = self.segment.index.min()
        self.right_segment_date = self.segment.index.max()
        self.features = features
        self.has_events = any(self.get_label_tulp())
    
    def check_segment_event(self, event: str):
        # Use focused for identification to prevent useless events on borders
        left_focused_segment_date = self.segment_focus.index.min()
        right_focused_segment_date = self.segment_focus.index.max()
        if len(self.timestamp[event]) > 0:
            results = [is_date_in_range(event_date, left_focused_segment_date, right_focused_segment_date) for event_date in self.timestamp[event]]
            return any(results)
        return False
            
    def get_label(self):
        swarming = self.check_segment_event('swarming')
        feeding = self.check_segment_event('feeding')
        treatment = self.check_segment_event('treatment')
        honey = self.check_segment_event('honey')
        died = False
        wakeup = False
          
        if len(self.timestamp['died']) > 0:
            died = True
            
        if 'wakep' in self.timestamp.keys() and len(self.timestamp['wakeup']) > 0:
            wakeup = True
        
        return {
            'swarming': 1 if swarming else 0,
            'feeding': 1 if feeding else 0,
            'died': 1 if died else 0,
            'treatment': 1 if treatment else 0,
            'honey': 1 if honey else 0,
            'wakeup': 1 if wakeup else 0
        }
        
    def get_label_tulp(self):
        label = self.get_label()
        return [label[e] for e in self.events]
    
    def get_data(self) -> pd.DataFrame:
        return self.segment[self.features]
    
    def show_segment(self):
        events_dict = {
            'Swarming': [pd.Timestamp(ts) for ts in self.timestamp['swarming']],
            'Feeding': [pd.Timestamp(ts) for ts in self.timestamp['feeding']],
            'Treatment': [pd.Timestamp(ts) for ts in self.timestamp['treatment']],
            'Honey': [pd.Timestamp(ts) for ts in self.timestamp['honey']],
            'Died': [pd.Timestamp(ts) for ts in self.timestamp['died']],
            'Wakeup': [pd.Timestamp(ts) for ts in self.timestamp['wakeup']],
        }
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8), sharex=True)
        fig.suptitle(
            f'Segment - {self.timestamp['year']}_{self.timestamp['hive_number']}\n{self.left_segment_date.strftime("%Y-%m-%d")} to {self.right_segment_date.strftime("%Y-%m-%d")}', 
            fontsize=16
        )
        ax1.plot(self.segment.index, self.segment['temp_mid'], label='temp_mid')
        ax1.plot(self.segment.index, self.segment['t_o'], label='t_o')
        ax1.set_ylabel('Temperature (°C)')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.segment.index, self.segment['weight_kg_smoothed'], label='weight_kg_smoothed')
        ax2.set_ylabel('Weight (kg)')
        ax2.legend()
        ax2.grid(True)
        
        ax3.plot(self.segment.index, self.segment['temp_ratio'], label='temp_ratio')
        ax3.set_ylabel('Temp ratio (IN/OUT)')
        ax3.legend()
        ax3.grid(True)
        
        ax4.plot(self.segment.index, self.segment['weight_kg_3_pct'], label='weight_kg_3_pct')
        ax4.set_ylabel('Weight 3 Percent Change')
        ax4.legend()
        ax4.grid(True)
        
        print(events_dict)

        plt.tight_layout()
        plt.show()

class BeeTimestamp:
    def __init__(self, timestamp, dir: str, features: list[str] = [], events: list[str] = [], step: int = 1):
        self.timestamp = timestamp
        self.dir = dir
        self.features = features
        self.events = events
        self.step = step
        
    def get_file(self) -> pd.DataFrame:
        filename = get_filename_from_timestamp(self.timestamp)
        return dataset.read_dataset_file(os.path.join(self.dir, filename))
    
    def get_segments(self, df: pd.DataFrame, segment_size: int = 12) -> list[BeeSegment]:
        segments = []
        for i in range(0, len(df), self.step):
            if i + segment_size <= len(df):
                s = BeeSegment(
                    timestamp=self.timestamp, 
                    segment=df.iloc[i:i+segment_size], 
                    features=self.features, 
                    events=self.events,
                    focus_margin=FOCUS_MARGIN
                )
                segments.append(s)
        return segments
    
    def get_segments_full(self, segment_size: int = 12) -> list[BeeSegment]:
        df = self.get_file()
        return self.get_segments(df, segment_size)
    
    def get_segments_around_event(self, event: str, segment_size: int = 12, hours: int = 24):
        df = self.get_file()
        event_dates = self.timestamp[event]
        event_segments = []
        for event_date in event_dates:
            date_from: pd.Timestamp = (pd.to_datetime(event_date) - pd.Timedelta(hours=hours))
            date_to: pd.Timestamp = (pd.to_datetime(event_date) + pd.Timedelta(hours=hours/2))
            sliced_df = df.loc[date_from:date_to]
            segments = self.get_segments(sliced_df, segment_size)
            event_segments.extend(segments)
        return event_segments
    
    def get_segments_around_events(self, segment_size: int = 12, hours: int = 24):
        segments = []
        for event in self.events:
            if event in self.timestamp.keys() and len(self.timestamp[event]) > 0:
                segments.extend(self.get_segments_around_event(event, segment_size, hours))
        return segments
        
class BeeTimestamps:
    def __init__(self, timestamps_file: str, dir: str, features: list[str] = [], events: list[str] = [], step: int = 1):
        self.timestamps = get_all_timestamps(timestamps_file)
        self.dir = dir
        self.features = features
        self.events = events
        self.step = step
        
    def get_timestamps(self):
        return self.timestamps
    
    def get_stamps_with_event(self, event):
        return [stamp for stamp in self.timestamps if len(stamp[event]) > 0]
    
    def get_filenames_with_event(self, event):
        return [get_filename_from_timestamp(stamp) for stamp in self.get_stamps_with_event(event)]
    
    def get_processed_timestamps(self):
        return [BeeTimestamp(stamp, self.dir, self.features, self.events, self.step) for stamp in self.timestamps]
    
    def get_segments_for_minor_events(self, segment_size: int = 12, hours: int = 24):
        segments = []
        for bee_timestamp in self.get_processed_timestamps():
            # segments = bee_timestamp.get_segments_full(segment_size)
            segments.extend(
                bee_timestamp.get_segments_around_events(segment_size=segment_size, hours=hours)
            )
        return segments

# def balance_segments(segments: list[BeeSegment], events: list[str] = ['swarming', 'feeding', 'honey']):
#     # Get segments for each category
#     swarming_segments = [s for s in segments if s.get_label()['swarming'] == 1]
#     honey_segments = [s for s in segments if s.get_label()['honey'] == 1]
#     feeding_segments = [s for s in segments if s.get_label()['feeding'] == 1]
    
#     no_event_segments = [s for s in segments if all(s.get_label()[e] == 0 for e in events)]

#     # Find minimum count to balance all categories
#     min_count = min(
#         len(swarming_segments),
#         # len(honey_segments), 
#         len(feeding_segments),
#         len(no_event_segments)
#     )

#     # Take equal number of segments from each category
#     balanced_segments = []
#     balanced_segments.extend(swarming_segments[:min_count])
#     balanced_segments.extend(honey_segments[:min_count])
#     balanced_segments.extend(feeding_segments[:min_count])
#     balanced_segments.extend(no_event_segments[:min_count])

#     return balanced_segments

def balance_segments(segments: list[BeeSegment], events: list[str] = ['swarming', 'feeding', 'honey'], min_count: int = 100):
    # Get segments for each category
    event_segments = list()
    for event in events:
        event_segments.append([s for s in segments if s.get_label()[event] == 1])    
    event_segments.append([s for s in segments if all(s.get_label()[e] == 0 for e in events)])
    counts = [len(e) for e in event_segments]

    # Find minimum count to balance all categories
    min_count = max([min_count, min(counts)])

    # Take equal number of segments from each category
    balanced_segments = []
    for es in event_segments:
        balanced_segments.extend(es[:min_count])

    return balanced_segments

def randomize_segments(segments: list[BeeSegment], seed: int = 0):
    s = segments.copy()
    random.seed(seed)
    random.shuffle(s)
    return s