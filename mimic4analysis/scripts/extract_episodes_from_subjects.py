from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import sys
from tqdm import tqdm

from mimic4analysis.subject import read_stays, read_diagnoses, read_events, get_events_for_stay,\
    add_hours_elpased_to_events
from mimic4analysis.subject import convert_events_to_timeseries, get_first_valid_from_timeseries
from mimic4analysis.preprocessing import read_itemid_to_variable_map, map_itemids_to_variables, clean_events
from mimic4analysis.preprocessing import assemble_episodic_data


parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
parser.add_argument('--variable_map_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/itemid_to_variable_map_mimic_4.csv'),
                    help='CSV containing ITEMID-to-VARIABLE map.')
parser.add_argument('--reference_range_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/variable_ranges.csv'),
                    help='CSV containing reference ranges for VARIABLEs.')
args, _ = parser.parse_known_args()

var_map = read_itemid_to_variable_map(args.variable_map_file) # itemid, VARIABLE, MIMIC_LABEL

variables = var_map.VARIABLE.unique()

for subject_dir in tqdm(os.listdir(args.subjects_root_path), desc='Iterating over subjects'):
    dn = os.path.join(args.subjects_root_path, subject_dir)
    # print("dn: ", dn, "\nsubject_dir: ", subject_dir)
    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except:
        continue

    try:
        # reading tables of this subject
        stays = read_stays(os.path.join(args.subjects_root_path, subject_dir))
        diagnoses = read_diagnoses(os.path.join(args.subjects_root_path, subject_dir))
        events = read_events(os.path.join(args.subjects_root_path, subject_dir))
    except Exception as e:
        print(str(e))
        sys.stderr.write('Error reading from disk for subject: {}\n'.format(subject_id))
        continue

    episodic_data = assemble_episodic_data(stays, diagnoses)

    # cleaning and converting to time series
    events = map_itemids_to_variables(events, var_map)
    # print("before clean: \n", events)
    events = clean_events(events)
    #print("after clean: \n", events)
    if events.shape[0] == 0:
        # no valid events for this subject
        continue
    timeseries = convert_events_to_timeseries(events, variables=variables)
    # print("timeseries.columns: ", timeseries.columns, timeseries['charttime'])
    
    # extracting separate episodes
    for i in range(stays.shape[0]):
        stay_id = stays.stay_id.iloc[i]
        intime = stays.intime.iloc[i]
        outtime = stays.outtime.iloc[i]

        episode = get_events_for_stay(timeseries, stay_id, intime, outtime)
        if episode.shape[0] == 0:
            # no data for this episode
            continue
        
        episode = add_hours_elpased_to_events(episode, intime).set_index('hours').sort_index(axis=0)
        if stay_id in episodic_data.index:
            episodic_data.loc[stay_id, 'weight'] = get_first_valid_from_timeseries(episode, 'weight')
            episodic_data.loc[stay_id, 'height'] = get_first_valid_from_timeseries(episode, 'height')
        episodic_data.loc[episodic_data.index == stay_id].to_csv(os.path.join(args.subjects_root_path, subject_dir,
                                                                              'episode{}.csv'.format(i+1)),
                                                                 index_label='icustay')
        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "hours" else x))
        episode = episode[columns_sorted]
        episode.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}_timeseries.csv'.format(i+1)),
                       index_label='Hours')
