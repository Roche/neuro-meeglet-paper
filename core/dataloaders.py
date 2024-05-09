import h5io
import pandas as pd
import numpy as np
import mne

from pathlib import Path
from tqdm import tqdm
from mne_bids import find_matching_paths
from pyriemann.utils.base import nearest_sym_pos_def

from braindecode.datasets import WindowsDataset, BaseConcatDataset
from skorch.helper import SliceDataset


class DataScaler(object):
    """On call multiply x with scaling_factor. Useful e.g. to transform from V to uV."""
    def __init__(self, scaling_factor):
        self.scaling_factor = scaling_factor

    def __call__(self, x):
        return x * self.scaling_factor


def load_raw_data(io_params, target_name, filter_func=None, sample=None):
    """"
    Load data in raw format, i.e. from mne epochs.
    """

    # Load participants.tsv file
    df_subjects = pd.read_csv(io_params['bids_root'] / "participants.tsv", sep='\t')
    df_subjects = df_subjects.set_index('participant_id').sort_index()  # Sort rows by participant_id

    if filter_func is not None:
        df_subjects = filter_func(df_subjects)
    
    if sample is not None:
        df_subjects = df_subjects.sample(sample, random_state=42)
    
    # Extract BIDS paths for all _epo.fif files in the derivative
    bids_paths = find_matching_paths(root=io_params["deriv_root"], suffixes="epo", extensions=".fif")
    bids_paths = [bp for bp in bids_paths if 'sub-' + bp.subject in df_subjects.index]

    # Create WindowsDatasets for each recording
    windows_datasets = []
    for i, bp in tqdm(enumerate(bids_paths, start=1), "Creating WindowsDatasets from epoch files"):
        # Read the epochs data
        epochs = mne.read_epochs(bp, preload=True)

        # Add target information as metadata
        target = df_subjects.loc[df_subjects.index == 'sub-' + bp.subject, target_name].values[0]
        if pd.isna(target):
            print(f"No target ({target}) information for subject {bp.subject}. Recording not added to dataset.")
            continue

        epochs.metadata = pd.DataFrame({
            "i_window_in_trial": list(range(len(epochs))),
            "i_start_in_trial": len(epochs) * [-1],  # dummy values
            "i_stop_in_trial": len(epochs) * [-1],  # dummy values
            "target": len(epochs) * [target]
        })

        description = dict(
            fname = bp.fpath,
            subject = bp.subject,
            session = bp.session,
            rec = i  # record id required for custom cv splitting
        )

        # Create a WindowsDataset from the epochs
        windows_dataset = WindowsDataset(epochs, description, transform=DataScaler(scaling_factor=1e6))
        windows_dataset.subject = bp.subject
        windows_dataset.session = bp.session

        windows_datasets.append(windows_dataset)

    # Combine WindowsDatasets into a ConcatDataset
    concat_dataset = BaseConcatDataset(windows_datasets)

    # Convert to SliceDataset
    X = SliceDataset(concat_dataset, idx=0, indices=None)
    y = SliceDataset(concat_dataset, idx=1, indices=None)

    # Remove subjects without data from df_subjects
    df_subjects = df_subjects[df_subjects.index.isin("sub-" + X.dataset.get_metadata().subject.unique())]

    # Extract feature info
    n_channels, window_size = X[0].shape
    feature_info = dict(
        n_channels=n_channels,
        window_size=window_size
    )

    return X, y, df_subjects, feature_info


def load_cov_data(io_params, target, features, feature_params, filter_func, sample):
    """
    Load Covariance Data
    """

    # Load participants.tsv file
    df_subjects = pd.read_csv(io_params['bids_root'] / "participants.tsv", sep='\t')
    df_subjects = df_subjects.set_index('participant_id').sort_index()  # Sort rows by participant_id

    if sample is not None:
        df_subjects = df_subjects.sample(sample, random_state=42)
    if filt := filter_func:
        df_subjects = filt(df_subjects)

    # Load data
    data = {
        subject: h5io.read_hdf5(io_params['deriv_root'] / subject / io_params['fname'])[features]
        for subject in df_subjects.index if (io_params['deriv_root'] / subject / io_params['fname']).exists()
    }

    covs = [data[sub][0][feature_params["key_cov"]] for sub in df_subjects.index if sub in data]    
    frequencies = next(data[sub][1][feature_params["key_freq"]] for sub in df_subjects.index if sub in data)
    
    if feature_params["frequencies_last"]:  # Transpose if frequencies are last
        covs = [c.T for c in covs]

    data_rank = None
    if fp := feature_params:
        if slicer := fp.get('slice_freqs'):
            print('slicing frequencies')
            covs = [c[slicer, ...] for c in covs]     
            if features == "coffeine":
                freq_idx = list(range(len(frequencies)))[slicer]
                frequencies = {k: v for ii, (k, v) in enumerate(frequencies.items()) if ii in freq_idx}
            elif features == "meeglet":
                frequencies = frequencies[slicer]
        if picks := fp.get('pick_channels'):
            print('picking channels')
            covs = [c[:, picks, :][:, :, picks] for c in covs]
        if nearest := fp.get('nearest_spd'):
            print('apply nearest symmetric positive matrix regularization')
            covs = [nearest_sym_pos_def(c, reg=nearest['reg']) for c in covs]
        if c_rank := fp.get('channel_rank'):
            if callable(c_rank):
                data_rank = c_rank(covs[0][0])
            elif isinstance(c_rank, int):
                data_rank = c_rank

    covs = np.array(covs)
    
    frequency_map = [(f"f{ii}", ii) for ii, _ in enumerate(frequencies)]
    freq_names = [name for name, _ in frequency_map]
    feature_info = dict(
        frequencies=frequencies,
        frequency_names=freq_names,
        data_rank=data_rank
    )

    df_subjects = df_subjects.loc[data.keys()]
    X = pd.DataFrame({name: list(covs[:, ii]) for name, ii in frequency_map})
    y = df_subjects[target].values

    # Remove bad subjects
    good_subjects = np.where(~np.isnan(y))
    X = X.iloc[good_subjects]
    y = y[good_subjects]
    df_subjects = df_subjects.iloc[good_subjects]

    return X, y, df_subjects, feature_info