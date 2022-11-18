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





def process_partition(args, partition, sample_rate=1.0, shortest_length=4.0, eps=1e-6):
    output_dir = os.path.join(args.output_path, partition)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    res_group = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))

    imv_file = os.path.join('/home/xu/dev/research/mimic4-analyze/data_prepare', 'mimic-invasive-ventilation_duration_min_12h.csv')

    imv_data = pd.read_csv(imv_file)

    print(imv_data.head())

    all_patients_meta_file = os.path.join('/home/xu/dev/research/mimic4-analyze/data_prepare', 'mimic-iv-patient-16-89.csv')

    all_patients_meta_data = pd.read_csv(all_patients_meta_file)

    # patient age range 16-89
    all_patients_qualified = all_patients_meta_data['subject_id'].values.tolist()

    cnt  = 0
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        
        # print("------------------------------------------------------------------------")
        # print("patient: ", patient, type(patient))
        # cnt += 1
        # if cnt == 5:
        #     break
        # if patient not in ['13927020']:
        #     continue
        # print("patient: ", patient)
        patient_folder = os.path.join(args.root_path, partition, patient)
        
        

        # patient_index = np.where(all_patients_qualified == int(patient))

        if int(patient) not in all_patients_qualified:
            continue

        # print('patient_folder: ', patient_folder)
        # procedures_iv = pd.read_csv(os.path.join(patient_folder, 'procedures.csv'))
        # print("procedures_iv.size: ", procedures_iv.shape[0])
        
        def procedures_iv_data_format(row):
            if row['valueuom'] == 'min':
                row['value'] = row['value'] / 60
                row['valueuom'] = 'hour'
            elif row['valueuom'] == 'day':
                row['value'] = row['value'] * 24
                row['valueuom'] = 'hour'
            return row
        
       
        # procedures_iv = procedures_iv.apply(lambda row: procedures_iv_data_format(row), axis = 1)
        
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        print('patient_ts_files: ', patient_ts_files)
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

                # stay_idx
                stay_idx = int(lb_filename[-5: -4]) - 1 # retrieve index from episode3.csv, the index is 2
                
                stays = pd.read_csv(os.path.join(patient_folder, 'stays.csv'))
                current_stay = stays.iloc[stay_idx]
                cur_stay_intime = pd.to_datetime(current_stay['intime'])
                cur_stay_outtime = pd.to_datetime(current_stay['outtime'])

                cur_stay_id = current_stay['stay_id']


                cur_imvs = imv_data[imv_data['stay_id'] == cur_stay_id]

                if cur_imvs.shape[0] == 0: # no imv 
                    continue
                
                print(".................\n", cur_imvs, '->',cur_imvs.shape[0])
                
                # procedures_iv, get the procedures in this stay
                iv_to_keep = []
                for i in range(cur_imvs.shape[0]):
                    p = cur_imvs.iloc[i]
                    p_starttime = pd.to_datetime(p['starttime'])
                    p_endtime = pd.to_datetime(p['endtime'])
                    if p_endtime <= cur_stay_intime or p_starttime >= cur_stay_outtime:
                        continue
                    iv_to_keep.append(i)


                if len(iv_to_keep) < 1:
                    continue
            

                first_iv = cur_imvs.iloc[iv_to_keep[0]]

                first_iv_starttime = pd.to_datetime(first_iv['starttime'])
                diff_in_first_iv_and_stay = pd.Timedelta(first_iv_starttime - cur_stay_intime) / np.timedelta64(1, 'h')
                diff_in_first_iv_and_stay = math.floor(diff_in_first_iv_and_stay)
               
                shortest_length = diff_in_first_iv_and_stay if diff_in_first_iv_and_stay > 0 else 0

                print('shortest_length: ', shortest_length)
                
                procedures_iv_to_keep = cur_imvs.iloc[iv_to_keep]
               
                # traverse all extubations
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
                
                
                # print("extubation_failed: ", extubation_failed)
                

                # check the last extubation
                ivm_session_len = procedures_iv_to_keep.shape[0]
                last_ivm_session = procedures_iv_to_keep.iloc[ivm_session_len - 1]
                # print('>>>', last_ivm_session)

                '''
                Check the last extubation. If the patient died within 48 hours after extubation, 
                we'll treat this extubation as failed. This aligs the paper: 
                '''
                # get stay info
                # use current_stay
                
                if current_stay['mortality'] or current_stay['mortality_inunit']: # if the patient died
                    death_time = pd.to_datetime(current_stay['deathtime'])
                    last_extubation_time = pd.to_datetime(last_ivm_session['endtime'])
        
                    if pd.Timedelta(death_time - last_extubation_time) / np.timedelta64(1, 'h') <= 48:
                        extubation_failed[len(extubation_failed) - 1] = True



                total_time_iv = 0
                invasive_ventilation = []
                for i in range(procedures_iv_to_keep.shape[0]):
                    iv =procedures_iv_to_keep.iloc[i]
                    iv_start = pd.to_datetime(iv['starttime'])
                    iv_end = pd.to_datetime(iv['endtime'])
                    interval = [pd.Timedelta(iv_start - cur_stay_intime) / np.timedelta64(1, 'h'), pd.Timedelta(iv_end - cur_stay_intime) / np.timedelta64(1, 'h')]
                    invasive_ventilation.append(interval)
                    total_time_iv += (interval[1] - interval[0])
                    
                
                
                # print("total_time_iv: ", total_time_iv, invasive_ventilation)


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
                # sample_times = np.arange(0.0, los + eps, sample_rate)
                # print("sample_times: ", sample_times)
                # print("shortest_length: ", shortest_length)
                # sample_times = list(filter(lambda x: x > shortest_length, sample_times))

                # print("sample_times2: ", sample_times)
                # # At least one measurement
                # sample_times = list(filter(lambda x: x > event_times[0], sample_times))
                
                # # print("sample_times3: ", sample_times)
                
                # output_ts_filename = patient + "_" + ts_filename
                # print("output_ts_filename: ", output_ts_filename)

                
                header = header[0:-1] + ',' + 'on_imv_cur_session' + ',' + 'on_imv_so_far' + '\n'
                on_imv_so_far_before_this_session = 0
                for idx, iv in enumerate(invasive_ventilation):
                    output_ts_filename = patient + "_" + str(idx) + "_" + ts_filename

                    iv_start = iv[0]
                    iv_end = iv[1]
                    with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                        outfile.write(header)
                        prev_ts = ''
                        for line in ts_lines:
                            # print("line: ", line)
                            line_ts = line.split(",")[0]
                            if line_ts == prev_ts:
                                continue
                            prev_ts = line_ts
                            line_ts_float = float(line_ts)
                            if (iv_start - eps) <= line_ts_float and line_ts_float <= (iv_end + eps):
                                on_imv_cur_session = line_ts_float - iv_start
                                on_imv_so_far = on_imv_so_far_before_this_session + on_imv_cur_session
                
                                line = line[0:-1] + ',' + str(round(on_imv_cur_session, 3)) + ',' + str(round(on_imv_so_far, 3)) + '\n'
                                outfile.write(line)

                    on_imv_so_far_before_this_session += (iv_end - iv_start)

                    ef = 1 if extubation_failed[idx] else 0
                    res_group.append((output_ts_filename, ef))
                

                # for idx, iv in enumerate(invasive_ventilation):
                #     print(idx, '->', iv, '->', extubation_failed[idx])
                #     iv_start = math.floor(iv[0])
                #     iv_end = math.ceil(iv[1])
                #     ef = 1 if extubation_failed[idx] else 0
                    
                #     res_group.append((output_ts_filename, iv_start, iv_end, ef))


    if partition == "train":
        # random.shuffle(res_group)
        res_group = sorted(res_group)
    if partition == "test":
        res_group = sorted(res_group)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,y_true\n')
        for (x, y) in res_group:
            listfile.write('{},{:}\n'.format(x, y))




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
