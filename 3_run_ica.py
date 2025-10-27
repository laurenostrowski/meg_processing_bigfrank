#### Run ICA

# Use ICA to remove ocular and cardiac artifacts from the concatenated, Maxwell-filtered MEG data. After loading `*_ALL_sss_raw.fif`, bad channels from `{subject}_bads.txt` are excluded, data are bandpass filtered 1–200 Hz and 60 Hz-notch filtered, then ICA is fit on MEG channels. Artifact components are identified automatically by correlating component time courses with EOG (blinks/eye movements) and ECG (heartbeat), those components are excluded, and the cleaned data are reconstructed and saved as `{subject}_ALL_post_ica-raw.fif` (the ICA solution is also saved to `{subject}_ALL-ica.fif`). Below the automatic pipeline, you can optionally review component topographies/time courses and adjust exclusions manually before applying.

# *The final ICA-cleaned data saved at `{subject}_ALL_post_ica-raw.fif` should be used for all analyses going forward.*

# load packages
# load packages
import numpy as np
import os
import time
import mne
from mne.io import read_raw_fif
from mne.preprocessing import ICA, read_ica

# subject info
subject = 'nbl_010'
ses = '01'

# filter params
filt_l, filt_h = 1, 200
filt_notch = 60

# ICA params
decim = 5
reject = dict(mag=5e-12, grad=4e-10)
random_state = 42
max_iter = 10000

# file paths
meg_dir = os.path.join('/mnt/sphere/nbl/processed_meg/', subject, 'ses-'+ses)
raw_fname = os.path.join(meg_dir, f"{subject}_ALL_sss_raw.fif")
er_fname = os.path.join(meg_dir, f"{subject}_emptyroom_sss_raw.fif")
bads_list_fname = os.path.join(meg_dir, f"{subject}_bads.txt")
ica_solution_fname = os.path.join(meg_dir, f"{subject}_ALL-ica.fif")
ica_processed_fname = os.path.join(meg_dir, f"{subject}_ALL_post_ica-raw.fif")  # ICA applied to raw data

if not os.path.exists(bads_list_fname):
    raise FileNotFoundError(f"Bad channels file not found: {bads_list_fname}. Go back and make it.")
with open(bads_list_fname, 'r') as f: bads = [line.strip() for line in f.readlines()]

if os.path.exists(ica_processed_fname):
    print('ICA already complete for %s session %s.'%(subject, ses))
else:
    ### SECTION 1: Fit the ICA solution ###
    print(f"Reading raw file for {subject}...")
    t = time.time()
    raw = read_raw_fif(raw_fname, preload=True, verbose=True)
    elapsed_readraw = time.time() - t
    print(f"Loaded in {elapsed_readraw:.1f} seconds.\n")

    print("Excluding bad channels...")
    raw.info['bads'] = bads
    print('Bad channels:', bads)
    raw.pick(picks='all', exclude='bads')
    raw.info.normalize_proj()

    print("Filtering data...")
    raw = raw.filter(filt_l, filt_h, picks='meg', n_jobs=-1)
    raw = raw.notch_filter(filt_notch, picks='meg', n_jobs=-1)
    
    if os.path.exists(ica_solution_fname):
        print("Loading existing ICA solution…")
        ica = mne.preprocessing.read_ica(ica_solution_fname)
    else:
        print("Fitting ICA...")
        ica = ICA(n_components=0.95, method='picard', max_iter=max_iter, random_state=random_state)
        t = time.time()
        ica.fit(raw, decim=decim, reject=reject, picks='meg')
        elapsed_ica = (time.time() - t)/60
        print("ICA fit elapsed time in minutes: %s" %elapsed_ica)

    ### SECTION 2: Apply the ICA solution ###                
    # EOG and ECG adjustments
    ica.exclude = []
    # find which ICs match the EOG and ECG pattern
    print("Finding EOG scores...")
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='EOG001')
    print("Finding ECG scores...")
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG002')

    eog_ecg_indices = eog_indices + ecg_indices
    ica.exclude = eog_ecg_indices

    print("Saving ICA solution...")
    ica.save(ica_solution_fname, overwrite=True)  # save solution with automatic exclusions

    # apply the ICA solution to the raw data and save
    print("Applying ICA...")
    ica.apply(raw)
    print("Saving ICA-processed data...")
    raw.save(ica_processed_fname, overwrite=True)
    print(f"ICA applied and saved for {subject}\n\n")
