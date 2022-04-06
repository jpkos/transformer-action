# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 19:44:54 2021

@author: jankos
"""
import pandas as pd
import cv2
#%% Get annotations
cols = ['annotation_type', 'time_min1', 'time_sec1', 'time_ms1',
                            'time_min2', 'time_sec2', 'time_ms2',
                            'time_min3', 'time_sec3', 'time_ms3', 'action']
anno_df = pd.read_csv('drilling.txt', header=None,
                      sep=r'\t+', names=cols, engine='python')
#%% Get eye tracker data
eye_df = pd.read_csv('EyeData_0.csv', sep=';')
eye_df['time_ms'] = eye_df[' Time Stamp '] - eye_df[' Time Stamp '].iloc[0]
#Segment drilling/not drilling
eye_df['action'] = pd.cut(eye_df['time_ms'], bins=anno_df['time_ms1'],
                          labels=anno_df['action'].iloc[:-1], ordered=False)
eye_df['action'] = eye_df['action'].fillna('Not drilling')
#%%Create video
video_name = 'test_annotations.mp4'
fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
vidwriter = cv2.VideoWriter(video_name, fourcc, 30, (640, 480))
missing_files = []
for i, row in eye_df.iterrows():
    frame_n = row[' Scene Picture Number ']
    frame_file = f'ScenePics_0/ScenePics_0/frame_{frame_n}.jpeg'
    try:#just in case the file does not exist
        frame = cv2.imread(frame_file)
    except FileNotFoundError:
        print(f'not found: {frame_file}')
        missing_files.append(frame_file)
        continue
    cv2.putText(frame, row['action'], (50,50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    vidwriter.write(frame)
    print(f'{i+1}/{len(eye_df)}', flush=True)
print(f'{len(missing_files)} files not found')    
cv2.destroyAllWindows()
vidwriter.release()
