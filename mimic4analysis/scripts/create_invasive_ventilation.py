from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd
import random
import math
random.seed(49297)
from tqdm import tqdm


def process_partition(args, partition, sample_rate=6.0, shortest_length=4.0, eps=1e-6):
    output_dir = os.path.join(args.output_path, partition)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    res_group = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        
        # print("------------------------------------------------------------------------")
        # print("patient: ", patient, type(patient))
        # cnt += 1
        # if cnt == 10:
        #     break
        # if patient not in ['15316288']:
        #     continue
        patient_folder = os.path.join(args.root_path, partition, patient)
        
    
        procedures_iv = pd.read_csv(os.path.join(patient_folder, 'procedures.csv'))
        # print("procedures_iv.size: ", procedures_iv.shape[0])
        
    
        def procedures_iv_data_format(row):
            if row['valueuom'] == 'min':
                row['value'] = row['value'] / 60
                row['valueuom'] = 'hour'
            elif row['valueuom'] == 'day':
                row['value'] = row['value'] * 24
                row['valueuom'] = 'hour'
            return row
        
       
        procedures_iv = procedures_iv.apply(lambda row: procedures_iv_data_format(row), axis = 1)
        
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                # print("lb_filename: ", lb_filename, patient_folder)
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))
                # empty label file
                if label_df.shape[0] == 0:
                    print("\n\t(empty label file)", patient, ts_filename)
                    continue
                
                los = 24.0 * label_df.iloc[0]['length of stay']  # in hours
                
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                stay_idx = int(lb_filename[-5: -4]) - 1 # retrieve index from episode3.csv, the index is 2
                
                stays = pd.read_csv(os.path.join(patient_folder, 'stays.csv'))
                current_stay = stays.iloc[stay_idx]
                cur_stay_intime = pd.to_datetime(current_stay['intime'])
                cur_stay_outtime = pd.to_datetime(current_stay['outtime'])
                
                # procedures_iv, get the procedures in this stay
                iv_to_keep = []
                for i in range(procedures_iv.shape[0]):
                    p = procedures_iv.iloc[i]
                    p_starttime = pd.to_datetime(p['starttime'])
                    p_endtime = pd.to_datetime(p['endtime'])
                    if p_endtime <= cur_stay_intime or p_starttime >= cur_stay_outtime:
                        continue
                    iv_to_keep.append(i)
                
                print("iv_to_keep: ", iv_to_keep, len(iv_to_keep))
                    
                
                if len(iv_to_keep) == 0:
                    continue
            
                first_iv = procedures_iv.iloc[iv_to_keep[0]]
               
                first_iv_starttime = pd.to_datetime(first_iv['starttime'])
                diff_in_first_iv_and_stay = pd.Timedelta(first_iv_starttime - cur_stay_intime) / np.timedelta64(1, 'h')
                diff_in_first_iv_and_stay = math.floor(diff_in_first_iv_and_stay)
               
                shortest_length = diff_in_first_iv_and_stay if diff_in_first_iv_and_stay > 0 else 0
                
                procedures_iv_to_keep = procedures_iv.iloc[iv_to_keep]
                print("procedures_iv_to_keep\n", procedures_iv_to_keep)
                
                extubation_failed = [False] * len(iv_to_keep)
                for i in range(procedures_iv_to_keep.shape[0] - 1):
                    iv_i = procedures_iv_to_keep.iloc[i]
                    iv_i_1 = procedures_iv_to_keep.iloc[i + 1]
                    
                    iv_i_endtime = pd.to_datetime(iv_i['endtime'])
                    iv_i_1_starttime = pd.to_datetime(iv_i_1['starttime'])
                    
                    test = pd.Timedelta(iv_i_1_starttime - iv_i_endtime) / np.timedelta64(1, 'h')
                    # print(test, type(test))
                    if pd.Timedelta(iv_i_1_starttime - iv_i_endtime) / np.timedelta64(1, 'h') <= 72:
                        extubation_failed[i] = True
                
                extubation_failed[-1] = False
                
                # print("extubation_failed: ", extubation_failed)
                
                total_time_iv = 0
                invasive_ventilation = []
                for i in range(procedures_iv_to_keep.shape[0]):
                    iv =procedures_iv_to_keep.iloc[i]
                    iv_start = pd.to_datetime(iv['starttime'])
                    iv_end = pd.to_datetime(iv['endtime'])
                    interval = [pd.Timedelta(iv_start - cur_stay_intime) / np.timedelta64(1, 'h'), pd.Timedelta(iv_end - cur_stay_intime) / np.timedelta64(1, 'h')]
                    invasive_ventilation.append(interval)
                    total_time_iv += (interval[1] - interval[0])
                    
                
                print("total_time_iv: ", total_time_iv, invasive_ventilation)
                # remove the total_time_iv < 24 hours
                if total_time_iv < 24:
                    continue

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                # print("event_time: ", event_times)
                # print("esp: ", eps, " los: ", los)
                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < los + eps]
                event_times = [t for t in event_times
                               if -eps < t < los + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                # print("event_times: ", event_times)
                sample_times = np.arange(0.0, los + eps, sample_rate)
                # print("sample_times: ", sample_times)
                # print("shortest_length: ", shortest_length)
                sample_times = list(filter(lambda x: x > shortest_length, sample_times))

                # print("sample_times2: ", sample_times)
                # At least one measurement
                sample_times = list(filter(lambda x: x > event_times[0], sample_times))
                
                # print("sample_times3: ", sample_times)
                
                output_ts_filename = patient + "_" + ts_filename
                # print("output_ts_filename: ", output_ts_filename)
                
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    prev_ts = ''
                    for line in ts_lines:
                        # print("line: ", line)
                        line_ts = line.split(",")[0]
                        if line_ts == prev_ts:
                            continue
                        prev_ts = line_ts
                        outfile.write(line)
                        
                def in_which_iv(t, ivs):
                    for i in range(len(ivs)):
                        iv = ivs[i]
                        if t >= iv[0] and t <= iv[1]:
                            return i
                    return -1

                for t in sample_times:
                    
                    t_in_iv_idx = in_which_iv(t, invasive_ventilation)
                    # print(t, "t_in_iv_idx: ", t_in_iv_idx)
                    
                    if len(invasive_ventilation) <= 1:
                        continue
                    if t_in_iv_idx == -1:
                        continue
                    
                    # removed the last invasive ventilation
                    # if t_in_iv_idx == len(invasive_ventilation) - 1:
                    #     continue
                    
                    # make sure that the time series data has at least 6 hour data to predict. 
                    t_in_iv = invasive_ventilation[t_in_iv_idx]
                    # print("t_in_iv: ", t_in_iv[0])
                    # print("t", t)
                    if t - t_in_iv[0] < 6:
                        continue
                    # print("t: ", t)
                    failed_ext = extubation_failed[t_in_iv_idx]
                    failed_ext_label = 1 if failed_ext else 0
                    
                    res_group.append((output_ts_filename, t, failed_ext_label))
                

    # print("Number of created samples:", len(res_group))
    if partition == "train":
        # random.shuffle(res_group)
        res_group = sorted(res_group)
    if partition == "test":
        res_group = sorted(res_group)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,y_true\n')
        for (x, t, y) in res_group:
            listfile.write('{},{:.6f},{:}\n'.format(x, t, y))


def main():
    parser = argparse.ArgumentParser(description="Create data for length of stay prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()
