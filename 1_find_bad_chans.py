# load packages
import os
import time
import mne
from mne.io import read_raw_fif

# session info
subject = 'nbl_011'
session = '01'
file = 'emptyroom' # bigfrank_1, bigfrank_2, bigfrank_3, emptyroom

# filter params
filt_l = 1
filt_notch = 60
filt_h = 200

# file paths
processed_meg_dir = '~/mnt/sphere/nbl/processed_meg/'
sub_ses = os.path.join(subject, 'ses-'+session)
recording_fname = f"{subject}_{file}_raw.fif"
raw_fname = os.path.join(processed_meg_dir, sub_ses, recording_fname)
bads_list_fname = os.path.join(processed_meg_dir, sub_ses,'%s_bads.txt'%(subject))

print (f"Reading raw file for {subject}...")
t = time.time()
raw = read_raw_fif(raw_fname, preload=True, verbose=True)
elapsed_readraw = time.time() - t
print(f"Loaded in {elapsed_readraw:.1f} seconds.\n")
print ("Filtering data...")
raw = raw.filter(filt_l, filt_h, n_jobs=-1)
raw = raw.notch_filter(filt_notch, n_jobs=-1)

# pick bad channels and annotate
raw.plot()

## AFTER you've finished reviewing and annotating the data, save out a list of the bad channels
with open(bads_list_fname, 'w') as f:
    for line in raw.info['bads']:
        f.write(f"{line}\n")