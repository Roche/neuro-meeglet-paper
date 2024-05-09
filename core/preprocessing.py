import numpy as np
import pandas as pd
import mne
import mne_bids
import mne_icalabel
import h5io
import autoreject
import coffeine
import meeglet

from pathlib import Path
from types import SimpleNamespace

def make_config(
    subject: str,
    bids_root: str | Path,
    deriv_root: str | Path,
    session: str,  # e.g. '1' or '01'
    tasks: list,  # can specify multiple tasks e.g. ['restEC', 'restEO'] or just one e.g. ['rest']
    run: str | None,
    analyze_channels: list,  # e.g. ['Fp1', 'Fp2', ...]
    notch_filter_freq: float,
    ic_rejection_criterion: str = "default",
    montage: str = 'standard_1005',
):
    return SimpleNamespace(
        # Dataset Params
        subject=subject,
        bids_root=bids_root,
        deriv_root=deriv_root,
        tasks=tasks,
        montage = montage,
        bids_path = mne_bids.BIDSPath(
            root=bids_root,
            subject=subject.lstrip('-sub'),
            session=session,
            task=tasks[0],
            run = run,
            datatype='eeg',
            extension='.vhdr'
        ),
        # Preprocessing Params
        crop=dict(tmin=0, tmax=900),
        resample=dict(sfreq=250),
        filter=dict(l_freq=1, h_freq=100),
        notch_filter=dict(freqs=(notch_filter_freq), method='spectrum_fit', filter_length='10s'),
        epochs=dict(duration=10, overlap=0, default_id=1, preload=True, event_id=None),
        analyze_channels = analyze_channels,
        ica = dict(
            method='picard',
            n_components=len(analyze_channels),
            keep_other=True,
            threshold=ic_rejection_criterion,
            random_state=20230226
        ),
        autoreject = dict(
            n_interpolate=np.array([1, 4, 8]),
            consensus=np.linspace(0, 1.0, 11),
            cv=5,
            random_state=20230226
        ),
        eeg_reference = dict(ref_channels='average'),
        # Feature Computation Params
        meeglet = dict(
            features=('cov'),
            foi_start=1,
            foi_end=64,
            bw_oct=0.5,
            delta_oct=None,
            density='oct'
        )
    )

def _read_concat_raw(cfg):
    # read and concat raw over tasks
    raw_list = list()
    for task in cfg.tasks:
        cfg.bids_path.update(task=task)
        raw = mne_bids.read_raw_bids(cfg.bids_path)
        raw.load_data()
        raw_list.append(raw)
    raw = mne.concatenate_raws(raw_list)
    # set task entitiy to None if multiple tasks were concatenated
    if len(cfg.tasks) > 1:
        cfg.bids_path.update(task=None)
    return raw

def read_raw_bids_root(
        cfg: SimpleNamespace  # The config for a given dataset/benchmark
    ) -> SimpleNamespace:  # The config with instance set to raw
    # Read and concatenate cleaned raw data from root path 
    cfg.bids_path.update(root=cfg.bids_root)
    cfg.inst = _read_concat_raw(cfg)
    return cfg

def read_raw_bids_derivative(
        cfg: SimpleNamespace  # The config for a given dataset/benchmark
    ) -> SimpleNamespace:  # The config with instance set to raw
    # Read and concatenate cleaned raw data from derivatives path 
    cfg.bids_path.update(root=cfg.deriv_root)
    cfg.inst = _read_concat_raw(cfg)
    return cfg

def crop(
        cfg: SimpleNamespace  # The config for a given dataset/benchmark
    ) -> SimpleNamespace:  # The config with instance cropped
    # Crop instance
    
    params_ = dict(cfg.crop)
    params_['tmax'] = min(params_['tmax'], cfg.inst.times[-1])
        
    cfg.inst.crop(**params_)
    return cfg

def make_epochs(
        cfg: SimpleNamespace  # The config for a given dataset/benchmark
    ) -> SimpleNamespace:  # The config with instance set to epochs
    # Compute epochs from raw
    epochs = mne.make_fixed_length_epochs(
        cfg.inst,
        duration=cfg.epochs['duration'],
        overlap=cfg.epochs['overlap'],
        id=cfg.epochs['default_id'],
        preload=cfg.epochs['preload']
    )
    if cfg.epochs['event_id'] is not None:
        epochs.event_id = cfg.epochs['event_id']
    cfg.inst = epochs
    return cfg

def set_channel_types(
        cfg: SimpleNamespace  # The config for a given dataset/benchmark
    ) -> SimpleNamespace:  # The config with updated instance
    # Update channel types according to selection
    cfg.inst.set_channel_types(cfg.channel_types_map)
    return cfg

def set_montage(
        cfg: SimpleNamespace  # The config for a given dataset/benchmark
    ) -> SimpleNamespace:  # The config with updated instance
    # Set EEG montage
    cfg.inst.set_montage(
        cfg.montage,
        match_case=False,
        on_missing='ignore'
    )
    return cfg

def map_artifact_ch_to_eeg(
        cfg: SimpleNamespace  # The config for a given dataset/benchmark
    ) -> SimpleNamespace:  # The config with instance updated
    cfg.inst.set_channel_types(
        dict(zip(cfg.analyze_channels, ['eeg'] * len(cfg.analyze_channels)))
    )
    return cfg

def select_channels(
        cfg: SimpleNamespace  # The config for a given dataset/benchmark
    ) -> SimpleNamespace:  # The config with instance updated
    # Select and reorder channels according to selection in config
    cfg.inst = cfg.inst.pick(cfg.analyze_channels)
    if cfg.inst.ch_names != cfg.analyze_channels:
        raise RuntimeError('Channel selection did not work. Please check the channel selection.')
    return cfg

def resample(
        cfg: SimpleNamespace  # The config for a given dataset/benchmark
    ) -> SimpleNamespace:  # The config with instance resampled
    cfg.inst.resample(**cfg.resample)
    return cfg

def filter(
        cfg: SimpleNamespace  # The config for a given dataset/benchmark
    ) -> SimpleNamespace:  # The config with instance filtered and resampled
    # Filter instance according to selection in config
    cfg.inst.filter(**cfg.filter)
    if hasattr(cfg, 'notch_filter'):
        cfg.inst.notch_filter(**cfg.notch_filter)
    return cfg

def _compute_ica_labels_reject(inst, ica, cfg):
    ica.fit(inst)
    pred_probas = mne_icalabel.iclabel.iclabel_label_components(
        inst, ica, inplace=True)

    df_label = pd.DataFrame(
        pred_probas,
        columns=[
            'brain', 'muscle artifact', 'eye blink',
            'heart beat', 'line noise', 'channel noise', 'other'
        ]
    )

    if cfg.ica['threshold'] == "default":
        # Reject components where the probability of at least one
        # artifact class exceeds the brain probability.
        artifact_labels = ['muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise']
        exclude = np.where(df_label["brain"] < df_label[artifact_labels].max(axis=1))[0]
        ica.exclude[:] = exclude
    elif cfg.ica['threshold'] == "keep_artifacts":
        df_label = df_label.loc[:, ['brain', 'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise']]  # ignore "other"
        exclude = np.where(df_label.idxmax(axis=1) == "brain")[0]
        ica.exclude[:] = exclude
    elif cfg.ica['threshold'] == "keep_ocular_artifacts":
        df_label = df_label.loc[:, ['brain', 'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise']]  # ignore "other"
        exclude = np.where(df_label.idxmax(axis=1) != "eye blink")[0]
        ica.exclude[:] = exclude
    elif cfg.ica['threshold'] == "keep_muscle_artifacts":
        df_label = df_label.loc[:, ['brain', 'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise']]  # ignore "other"
        exclude = np.where(df_label.idxmax(axis=1) != "muscle artifact")[0]
        ica.exclude[:] = exclude
    elif cfg.ica['threshold'] == "keep_other_artifacts":
        df_label = df_label.loc[:, ['brain', 'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise']]  # ignore "other"
        exclude = np.where(~df_label.idxmax(axis=1).isin(["line noise", "heart beat", "channel noise"]))[0]
        ica.exclude[:] = exclude
    else:
        raise ValueError('ICA threshold option not set or not known.')
    return ica.apply(inst, exclude=exclude)

def apply_ica(
        cfg: SimpleNamespace  # The config for a given dataset/benchmark
    ) -> SimpleNamespace:  # The config updated with ICA cleaned-instance + ICA object
    # Apply ICA cleaning
    ica = mne.preprocessing.ICA(
        method=cfg.ica['method'],
        n_components=cfg.ica['n_components'],
        random_state=cfg.ica['random_state']
    )
    _ica_fun = _compute_ica_labels_reject
    cfg.inst = _ica_fun(cfg.inst, ica, cfg)
    cfg.inst_ica = ica
    return cfg

def _fit_autoreject(epochs, ar):
    epochs_clean = ar.fit_transform(epochs)
    return epochs_clean

def compute_auto_reject(
        cfg: SimpleNamespace  # The config for a given dataset/benchmark
    ) -> SimpleNamespace:  # The config with autoreject-cleaned instance
    ar = autoreject.AutoReject(
        n_interpolate=cfg.autoreject['n_interpolate'],
        consensus=cfg.autoreject['consensus'],
        random_state=cfg.autoreject['random_state']
    )
    for attr in ('cv', 'random_state'):
        if hasattr(cfg.autoreject, attr):
            setattr(ar, getattr(cfg.autoreject, attr))
    _ar_fun = _fit_autoreject
    epochs_clean = _ar_fun(cfg.inst, ar)
    cfg.inst = epochs_clean
    return cfg

def set_eeg_reference(
        cfg: SimpleNamespace  # The config for a given dataset/benchmark
    ) -> SimpleNamespace:  # The config with EEG reference updated.
    # Set the EEG reference.
    cfg.inst.set_eeg_reference(**cfg.eeg_reference)
    return cfg

def _compute_coffeine(inst, cfg):
    features, info = coffeine.compute_features(
        inst,
        features=cfg.coffeine['features'],
        fs=inst.info['sfreq'],
        frequency_bands=cfg.coffeine['frequency_bands'])
    info['frequency_bands'] = cfg.coffeine['frequency_bands']
    return (features, info)

def compute_coffeine_features(
        cfg: SimpleNamespace  # The config for a given dataset/benchmark
    ) -> SimpleNamespace:  # The config with .features attribute set
    _feat_fun = _compute_coffeine
    if not hasattr(cfg, 'features'):
        cfg.features = {}
        
    cfg.features.update(coffeine = _feat_fun(cfg.inst, cfg))
    if conds := getattr(cfg, 'conditions', None):
        if isinstance(cfg.inst, (mne.Epochs, mne.EpochsArray)):
            results_by_condition = dict()
            for cond in conds:
                cond_key = cond.replace('/', '__')
                results_by_condition[cond_key] = _feat_fun(
                    cfg.inst[cond], cfg
                )
            cfg.features.update(coffeine_by_condition=results_by_condition)
    return cfg

def _compute_meeglet(inst, cfg):
    features = meeglet.compute_spectral_features(
        inst=cfg.inst,
        **cfg.meeglet 
    )
    return [vars(ff) for ff in features]

def compute_meeglet_features(
        cfg: SimpleNamespace  # The config for a given dataset/benchmark
    ) -> SimpleNamespace:  # The config with .features attribute set
    _feat_fun = _compute_meeglet
    if not hasattr(cfg, 'features'):
        cfg.features = {}
        
    cfg.features.update(meeglet = _feat_fun(cfg.inst, cfg))
    if conds := getattr(cfg, 'conditions', None):
        if isinstance(cfg.inst, (mne.Epochs, mne.EpochsArray)):
            results_by_condition = dict()
            for cond in conds:
                cond_key = cond.replace('/', '__')
                results_by_condition[cond_key] = _feat_fun(
                    cfg.inst[cond], cfg
                )
            cfg.features.update(meeglet_by_condition=results_by_condition)
    return cfg

def save_features(
        cfg: SimpleNamespace  # The config for a given dataset/benchmark
    ) -> SimpleNamespace: # Unmodified config, features saved
    out_path = cfg.deriv_root / cfg.subject
    out_path.mkdir(parents=True, exist_ok=True)
    h5io.write_hdf5(
        fname=out_path / 'features.h5',
        data=cfg.features,
        overwrite=True
    )
    return cfg

def save_epochs(
        cfg: SimpleNamespace  # Config with inst of type mne.Epochs
    ) -> SimpleNamespace: # Unmodified config, epochs saved
    
    if not isinstance(cfg.inst, (mne.Epochs, mne.EpochsArray)):
        raise TypeError("cfg.inst must be of type mne.Epochs")

    # Build out_path from bids path but replace root with deriv_root and adjust file name
    out_path = cfg.bids_path.copy().update(
        root=cfg.deriv_root, extension='.fif', suffix='epo', check=False).fpath
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save epochs
    if cfg.inst.times.flags['WRITEABLE'] is True:
        cfg.inst.times.flags['WRITEABLE'] = False
    cfg.inst.save(out_path, overwrite=True)
    return cfg

def get_common_c_picks(
        datasets: list, # the datasets
        channel_maps: dict # list of channels per dataset 
    ) -> dict:  # common indices per dataset
    # Compute common channel indices per dataset
    channels = [channel_maps[ds] for ds in datasets]
    common_channels = set.intersection(*map(set, channels))
    common_inds = [
        [chans.index(c) for c in common_channels] for chans in channels
    ]
    channels_selected = [
        [chans[ix] for ix in inds] for chans, inds in
        zip(channels, common_inds)
    ]
    for chans in channels_selected[1:]:
        assert (chans == channels_selected[0])
    return {ds: inds for ds, inds in zip(datasets, common_inds)}