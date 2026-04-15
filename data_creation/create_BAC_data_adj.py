import pandas as pd
import numpy as np
from Preprocessing.from_log_to_tensors import log_to_tensors
import os
import torch


def deduplicate_scans(df):
    """Remove consecutive duplicate scanner events within the same case.

    A duplicate is defined as an event with the same activity and location_id
    as the immediately preceding event in the same case, occurring within
    2 seconds. The first occurrence is kept.

    Returns
    -------
    pd.DataFrame
        Log with duplicate scanner events removed.
    """
    threshold_seconds = 2
    df = df.sort_values(['case:concept:name', 'time:timestamp']).reset_index(drop=True)
    prev_act = df.groupby('case:concept:name')['concept:name'].shift(1)
    prev_loc = df.groupby('case:concept:name')['location_id'].shift(1)
    prev_time = df.groupby('case:concept:name')['time:timestamp'].shift(1)
    time_diff = (df['time:timestamp'] - prev_time).dt.total_seconds()
    is_dup = (
        (df['concept:name'] == prev_act) &
        (df['location_id'] == prev_loc) &
        (time_diff <= threshold_seconds)
    )
    return df[~is_dup].reset_index(drop=True)


def prepare_bac_log(df):
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='mixed').dt.tz_convert('UTC')
    df['case:concept:name'] = df['case:concept:name'].astype('str')
    df['dep_gate_nbr'] = df['dep_gate_nbr'].astype('str')
    df['location_id'] = df['location_id'].astype('str')
    df['dep_flightnumb'] = df['dep_flightnumb'].astype('str')
    df = df.drop(['dep_final_destin', 'dap_screen_dep', 'dep_flightsuff', 'dep_scheduled_deptime'], axis='columns')
    df = deduplicate_scans(df)
    return df


def construct_BAC_adj_datasets():
    df = pd.read_csv('luggage_log.csv')
    df = prepare_bac_log(df)

    categorical_casefeatures = ['dep_airline', 'dep_airport', 'dep_flightnumb',
                                'dep_gate_nbr', 'dap_schen_ind', 'dap_geog_area']
    categorical_eventfeatures = ['location_id']
    numeric_casefeatures = []
    numeric_eventfeatures = []
    case_id = 'case:concept:name'
    timestamp = 'time:timestamp'
    act_label = 'concept:name'

    start_date = None
    end_date = None
    max_days = 0.07787037037037037
    window_size = 15
    log_name = 'BAC_adj'
    start_before_date = None
    test_len_share = 0.25
    val_len_share = 0.2
    mode = 'preferred'
    outcome = None

    result = log_to_tensors(df,
                            log_name=log_name,
                            start_date=start_date,
                            start_before_date=start_before_date,
                            end_date=end_date,
                            max_days=max_days,
                            test_len_share=test_len_share,
                            val_len_share=val_len_share,
                            window_size=window_size,
                            mode=mode,
                            case_id=case_id,
                            act_label=act_label,
                            timestamp=timestamp,
                            cat_casefts=categorical_casefeatures,
                            num_casefts=numeric_casefeatures,
                            cat_eventfts=categorical_eventfeatures,
                            num_eventfts=numeric_eventfeatures,
                            outcome=outcome)

    train_data, val_data, test_data = result

    output_directory = log_name
    os.makedirs(output_directory, exist_ok=True)

    # Save training tuples
    train_tensors_path = os.path.join(output_directory, 'train_tensordataset.pt')
    torch.save(train_data, train_tensors_path)

    # Save validation tuples
    val_tensors_path = os.path.join(output_directory, 'val_tensordataset.pt')
    torch.save(val_data, val_tensors_path)

    # Save test tuples
    test_tensors_path = os.path.join(output_directory, 'test_tensordataset.pt')
    torch.save(test_data, test_tensors_path)
