#### Perform Signal-Space Separation (SSS) and Maxwell filtering

# This step performs Maxwell filtering with tSSS (10 s window) on each run to suppress environmental noise, sensor cross-talk, and head-movement–related artifacts using site-specific calibration (`sss_cal.dat`) and cross-talk (`ct_sparse.fif`) files. Data from every run are transformed to a common head position (the first run’s `dev_head_t`) to standardize head geometry across runs. Each filtered run is saved (`_sss_raw.fif`). The filtered runs are then concatenated into a single continuous Raw object (`_ALL_sss_raw.fif`). *This is the file that should be fed into ICA in the next step.*

#### What's tSSS?
# Signal Space Separation (SSS) is a Maxwell-equation–based method that expands MEG sensor data into spherical-harmonic components originating inside vs. outside the helmet. By modeling the magnetic field, SSS retains “internal” brain sources and suppresses “external” interference; temporal SSS (tSSS) further rejects artifacts by removing components whose time courses match external sources across a sliding window. Here, SSS/tSSS are applied with site calibration and cross-talk files, data are transformed to the first run’s head position, each run is saved, then all runs are concatenated for analysis.

# *Requires unfiltered data and calibration files (sss_cal.dat, ct_sparse.fif)*

# load packages
import os
import time
import numpy as np
import mne
from mne.io import read_raw_fif
from mne.preprocessing import maxwell_filter

# session info
subject = 'nbl_010'
session = '01'
files = ['bigfrank_1', 'bigfrank_2', 'bigfrank_3', 'emptyroom'] # 'pharyperc'

# file paths
processed_meg_dir = '/mnt/sphere/nbl/processed_meg/'
sub_ses = os.path.join(subject, 'ses-'+session)
bads_list_fname = os.path.join(processed_meg_dir, sub_ses, '%s_bads.txt'%(subject))
with open(bads_list_fname, 'r') as f: bads = [line.strip() for line in f.readlines()]
print('Bad channels:',bads)
raw_fnames = [os.path.join(processed_meg_dir, sub_ses, subject+'_'+f+'_raw.fif') for f in files]

# Step 1: Load raw files and static head positions
raw_list = []

for idx, file in enumerate(files):
    fname = os.path.join(processed_meg_dir, sub_ses, f"{subject}_{file}_raw.fif")
    print(f"\n{'='*60}")
    print(f"Processing file {idx+1}/{len(files)}: {os.path.basename(fname)}")
    print('='*60)
    
    # load raw data
    print(f"Loading raw file...")
    t = time.time()
    raw = read_raw_fif(fname, preload=True, verbose=False)
    raw.info["bads"] = bads  # important!
    print(f"Loaded in {time.time() - t:.1f} seconds.\n")
    
    raw_list.append(raw)
    
    if file != 'emptyroom':
        print(f"File {idx+1} head position:")
        print(raw.info['dev_head_t']['trans'])

# Step 2: Maxwell filter
cal_fname = '/mnt/sphere/nbl/sss/sss_cal.dat'   # calibration file
ct_fname = '/mnt/sphere/nbl/sss/ct_sparse.fif'  # cross-talk file

# process each file with Maxwell filtering & transform to initial head position
destination = raw_list[0].info['dev_head_t']
raw_sss_list = []
for idx, (file, raw) in enumerate(zip(files, raw_list)):
    print(f"\n{'='*60}")
    print(f"Maxwell filtering file {idx+1}/3: {subject}_{file}_raw.fif")
    print(f"{'='*60}")
    print(f"Applying Maxwell filter with tSSS...")
    t = time.time()
    if file == 'emptyroom':
        raw_sss = maxwell_filter(raw, calibration=cal_fname, cross_talk=ct_fname, coord_frame='meg',
                                 destination=None, st_duration=10, origin='auto', verbose=True)
    else:
        raw_sss = maxwell_filter(raw, cross_talk=ct_fname, calibration=cal_fname, coord_frame='head',
                                 destination=destination, st_duration=10, origin=(0., 0., 0.04), verbose=True)
    elapsed = time.time() - t
    print(f"Maxwell filtering completed in {elapsed:.1f} seconds")
    if len(raw.ch_names) != len(raw_sss.ch_names):
        raise RuntimeError(f"Lost channels: Raw has {len(raw.ch_names)} channels; Raw_sss has {len(raw_sss.ch_names)} channels")
    
    if file == 'emptyroom':
        # bandpass filter now because the data won't be ICA-processed
        raw_sss = raw_sss.filter(1, 200, picks='meg', n_jobs=-1)
        raw_sss = raw_sss.notch_filter(60, picks='meg', n_jobs=-1)
    else:
        raw_sss_list.append(raw_sss)
    
    # save
    sss_fname = os.path.join(processed_meg_dir, sub_ses, f"{subject}_{file}_sss_raw.fif")
    raw_sss.save(sss_fname, overwrite=True)
    print(f"Saved to: {sss_fname}")

# concatenate all Maxwell filtered files
raw_concat = mne.concatenate_raws(raw_sss_list, preload=True)
print(f"Concatenated data:")
print(f"  Duration: {raw_concat.times[-1]:.1f} seconds ({raw_concat.times[-1]/60:.1f} minutes)")
print(f"  Samples: {len(raw_concat.times)}")
print(f"  Channels: {len(raw_concat.copy().pick('meg').ch_names)} MEG, ")

# save concatenated file
concat_fname = os.path.join(processed_meg_dir, sub_ses, f"{subject}_ALL_sss_raw.fif")
raw_concat.save(concat_fname, overwrite=True)
print(f"\nSaved concatenated file to: {concat_fname}")
