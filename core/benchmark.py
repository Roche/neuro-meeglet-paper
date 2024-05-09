from typing import Union, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
from copy import deepcopy

import itertools
import torch
import pandas as pd
import numpy as np

from braindecode.datasets import BaseConcatDataset
from skorch.helper import SliceDataset
from fastcore.transform import Pipeline

from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold, KFold, cross_val_predict
from sklearn.model_selection._split import _BaseKFold
from sklearn.metrics import r2_score, mean_absolute_error, balanced_accuracy_score, roc_auc_score, accuracy_score

from core.config import cfg
from core.dataloaders import load_raw_data, load_cov_data
from core.models import create_cov_model, create_raw_model


class Benchmark(ABC):
    """
    The `Benchmark` class encapsulates benchmark configurations along with the basic
    functionality to load and create the associated dataset and model.
    """

    def __init__(
        self,
        io_params: dict, # The bids_root and bids_derivative needed for this benchmark,
        dataset: str, # The name of the EEG dataset.
        processing: str, # The type of processing related to the deriv root.
        features: str, # The name of the type or class of features.
        model: str, # The name of the regression model.
        target: str, # The type of target.
        task: str = 'regression', # The type of learning task.
        feature_params: Union[dict, None]=None,
        filter_func: Union[callable, None]=None
    ): 
        if any(k not in io_params for k in ('bids_root', 'deriv_root', 'fname')):
            raise ValueError('We need the bids root and derivative paths.')
        
        self.io_params = io_params
        self.dataset = dataset
        self.processing = processing
        self.features = features
        self.model = model
        self.target = target
        self.task = task
        self.feature_params = feature_params
        self.filter_func = filter_func

    def get_params(self) -> dict:  # the config of the Benchmark
        """
        Get key coordinates for this benchmark.
        """
        return dict(
            dataset = self.dataset,
            processing = self.processing,
            features = self.features,
            model = self.model,
            target = self.target,
            task = self.task
        )

    def __repr__(self):
        params = '\n    '.join(
            [f'{k}: {v}' for k, v in self.get_params().items()])
        io = '\n    '.join(
            [f'{k}: {v}' for k, v in self.io_params.items()])
        out = f"<Benchmark:\n  Params:\n    {params}\n  I/O:\n    {io}>"
        return out

    @abstractmethod
    def load_data(self, sample=None):
        pass

    @abstractmethod
    def load_model(self, feature_info):
        pass

    def load(
        self,
        sample: int=None  # how many recordings to sample from the dataset (e.g., for testing / debugging purposes)
    ) -> Tuple[np.ndarray, np.ndarray, BaseEstimator, pd.DataFrame]:
        """
        Returns dataset and model. More specifically a tuple comprising: X, y, model, df_subjects.
        """
        X, y, df_subjects, feature_info = self.load_data(sample)
        model = self.load_model(feature_info)
        return X, y, model, df_subjects


class RawBenchmark(Benchmark):
    """
    `Benchmark` subclass to handle dataset and model creation for models operating directly on the raw,
    i.e. epoched data.
    """

    def load_data(self, sample=None):
        return load_raw_data(self.io_params, self.target, self.filter_func, sample)

    def load_model(self, feature_info):
        return create_raw_model(self.model, self.task, feature_info["n_channels"])


class CovBenchmark(Benchmark):
    """
    `Benchmark` subclass to handle dataset and model creation for models operating on covariance matrices.
    """
    
    def load_data(self, sample=None):
        return load_cov_data(self.io_params, self.target, self.features, self.feature_params, self.filter_func, sample)
    
    def load_model(self, feature_info):
        return create_cov_model(self.model, self.task, self.feature_params, feature_info)


def format_sex(df):
    df_out = df.copy()
    sex_map = {
        ('F', 'M'): (0, 1),  # embarc: F=female => 1=male
        (0, 1): (1, 0), # tdbrain: 1=female => 1=male
        (1, 2): (1, 0)  # tuab: 2=female => 1=male
    }
    
    if 'sex' not in df_out:
        if 'sex_x' in df_out.columns:
            df_out['sex'] = df_out['sex_x']
        elif 'gender'in df_out:
            df_out['sex'] = df_out['gender']
    
    sex_code = tuple(sorted(df_out['sex'].unique()))
    df_out['sex'] = df_out['sex'].map(dict(zip(sex_code, sex_map[sex_code])))
    
    return df_out


def tdbrain_add_indications(df_subjects):
    df_tab = df_subjects.pivot_table(
        values='age', columns=['indication'], aggfunc=[len]).T
    df_tab.columns = ['n_cases']
    df_tab['mean'] = df_subjects.pivot_table(
        values='age', columns=['indication'], aggfunc=[np.mean]).T.age.values
    df_tab = df_tab.sort_values('n_cases', ascending=False)
    df_tab = df_tab.reset_index().iloc[:, 1:]

    df_tab['diagnosis'] = df_tab.indication
    df_tab.loc[df_tab['n_cases'] < 20, 'diagnosis'] = 'other'

    df_subjects['diagnosis'] = df_subjects['indication'].map(
        dict(zip(df_tab['indication'], df_tab['diagnosis'])))
    df_subjects.loc[
        df_subjects.diagnosis.isna(), 'diagnosis'] = 'other'
    df_subjects['diagnosis_code'] = (
        df_subjects.diagnosis.astype('category').cat.codes)
    return df_subjects


select_tdbrain = Pipeline([tdbrain_add_indications, format_sex])
select_tuab = Pipeline([format_sex])


def rank_is_len_minus_one(x): return len(x) - 1


def create_benchmark_configs(datasets, models, features, processings, targets):
    param_list = itertools.product(datasets, models, features, processings, targets)
    benchmarks = list()
    for dataset, model, features, processing, target in param_list:

        # deep models only operate on raw data
        if model == 'shallow' and features != 'raw':
            continue
        
        # non-deep models only operate on precomputed features
        if features == 'raw' and model != 'shallow':
            continue
        
        io_params = dict(
            bids_root=Path(cfg["DATASETS"][dataset]["bids_root"]),
            deriv_root=Path(cfg["DATASETS"][dataset]["deriv_root"]) / processing,
            fname=f'features.h5',
        )

        # Specify rank to avoid riemann model violations due to rank deficiencies
        # Rank choosen through visual inspection of effective rank after ICA
        if model == "riemann" and processing == "preproc_autoreject_ica":
            channel_rank = 7
        elif model == "riemann" and processing in [
            "preproc_autoreject_ica_artifacts", "preproc_autoreject_ica_muscle_artifacts",
            "preproc_autoreject_ica_ocular_artifacts", "preproc_autoreject_ica_other_artifacts"
            ]:
            channel_rank = 2
        else:
            channel_rank = rank_is_len_minus_one

        if features == 'coffeine':
            feature_params = dict(
                channel_rank = channel_rank,
                nearest_spd = dict(reg=1e-15),  # value ad-hoc but conservative
                picks_channels = list(np.arange(19)) if dataset == 'tuab' else None,
                pca_reg = dict(reg = 1e-15),  # value ad-hoc but conservative
                slice_freqs = slice(0, 8, 1),
                key_cov = 'covs',
                key_freq = 'frequency_bands',
                frequencies_last = False
            )
        elif features == "meeglet":
            feature_params = dict(
                channel_rank = channel_rank,
                nearest_spd = dict(reg = 1e-15),  # value ad-hoc but conservative
                picks_channels = list(np.arange(19)) if dataset == 'tuab' else None,
                pca_reg=dict(reg = 1e-15),  # value ad-hoc but conservative,
                key_cov = 'cov',
                key_freq = 'foi',
                frequencies_last = True  # meeglet gives frequencies last
            )
        else:
            feature_params=None

        # Map targets to tasks
        task = {
            'sex': 'classification',
            'age': 'regression'
        }.get(target)

        # Filter functions to take care of special cases / dataset-dependent formats
        filter_func={
            "TDBRAIN": select_tdbrain,
            "TUAB": select_tuab
        }.get(dataset)

        if features == "raw":
            bm = RawBenchmark(
                io_params=io_params,
                dataset=dataset,
                processing=processing,
                features=features,
                model=model,
                target=target,
                task=task,
                feature_params=feature_params,
                filter_func=filter_func
            )
        elif (features == "coffeine") or (features == "meeglet"):
            bm = CovBenchmark(
                io_params=io_params,
                dataset=dataset,
                processing=processing,
                features=features,
                model=model,
                target=target,
                task=task,
                feature_params=feature_params,
                filter_func=filter_func
            )

        benchmarks.append(bm)

    return benchmarks


class StratifiedByGroupKFold(_BaseKFold):
    "Kfold with balanced proportions of grouping variables per fold (stratified by non-target)"

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self._cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    def _iter_test_indices(self, X, y=None, groups=None):
        for train, test in self._cv.split(X, y=groups, groups=None):
            yield test

    def split(self, X, y, groups=None):
        #Generate indices to split data into training and test set.
        if groups is None:
            raise ValueError("This CV scheme needs grouping information.")
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        return super().split(X, y, groups)
    

class BdecodeKFold(KFold):
    """An adapted sklearn.model_selection.KFold that gets skorch SliceDatasets
    holding braindecode datasets of length n_compute_windows but splits based
    on the number of original recording files."""
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def split(self, X, y=None, groups=None, yield_win_inds=True):
        """Generate indices to split data into training and test set.
        The split is done over the different datasets (i.e., recordings) in the
        provided SliceDataset(s), however by default the method yields indices
        to the windows of each of those datasets. To instead yield indices to
        the datasets, set `yield_win_inds` to False.
        Parameters
        ----------
        X : skorch.helper.SliceDataset
            Data to split. `X.dataset` must be a
            `braindecode.datasets.BaseConcatDataset`.
        y : skorch.helper.SliceDataset | None
            The targets to split.
        groups : array-like of shape (n_samples,) | None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        yield_win_inds : bool
            If True, yield indices to the windows of each recording. If False,
            instead yield indices to the datasets (i.e., recordings). This can
            be used to obtain a list of recordings used in each fold, which can
            be compared to per-recording KFold (e.g., as is done with non-DL
            benchmarks).
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        assert isinstance(X.dataset, BaseConcatDataset)
        assert isinstance(y.dataset, BaseConcatDataset)
        # split recordings instead of windows
        split = super().split(
            X=X.dataset.datasets, y=y.dataset.datasets, groups=groups)
        rec = X.dataset.get_metadata()[['rec']]
        rec['ind'] = rec.groupby('rec').ngroup()
        # the index of DataFrame rec now corresponds to the id of windows
        rec.reset_index(inplace=True, drop=True)
        for train_i, valid_i in split:
            if yield_win_inds:
                # map recording ids to window ids
                train_window_i = rec[rec['ind'].isin(train_i)].index.to_list()
                valid_window_i = rec[rec['ind'].isin(valid_i)].index.to_list()
                if set(train_window_i) & set(valid_window_i):
                    raise RuntimeError('train and valid set overlap')
                yield train_window_i, valid_window_i
            else:
                yield train_i, valid_i

class BdecodeStratifiedByGroupKFold(StratifiedByGroupKFold):
    """An adapted sklearn.model_selection.KFold that gets skorch SliceDatasets
    holding braindecode datasets of length n_compute_windows but splits based
    on the number of original recording files."""
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def split(self, X, y=None, groups=None, yield_win_inds=True):
        """Generate indices to split data into training and test set.
        The split is done over the different datasets (i.e., recordings) in the
        provided SliceDataset(s), however by default the method yields indices
        to the windows of each of those datasets. To instead yield indices to
        the datasets, set `yield_win_inds` to False.
        Parameters
        ----------
        X : skorch.helper.SliceDataset
            Data to split. `X.dataset` must be a
            `braindecode.datasets.BaseConcatDataset`.
        y : skorch.helper.SliceDataset | None
            The targets to split.
        groups : array-like of shape (n_samples,) | None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        yield_win_inds : bool
            If True, yield indices to the windows of each recording. If False,
            instead yield indices to the datasets (i.e., recordings). This can
            be used to obtain a list of recordings used in each fold, which can
            be compared to per-recording KFold (e.g., as is done with non-DL
            benchmarks).
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        assert isinstance(X.dataset, BaseConcatDataset)
        assert isinstance(y.dataset, BaseConcatDataset)

        # get grouping info on the recording level
        meta_data = X.dataset.get_metadata()[["rec"]]
        meta_data["group"] = groups
        groups = meta_data.drop_duplicates().sort_values("rec")["group"].values

        # split recordings instead of windows
        split = super().split(
            X=np.array(X.dataset.datasets, dtype=object),
            y=np.array(y.dataset.datasets, dtype=object),
            groups=groups
        )
        rec = X.dataset.get_metadata()[['rec']]
        rec['ind'] = rec.groupby('rec').ngroup()
        # the index of DataFrame rec now corresponds to the id of windows
        rec.reset_index(inplace=True, drop=True)
        for train_i, valid_i in split:
            if yield_win_inds:
                # map recording ids to window ids
                train_window_i = rec[rec['ind'].isin(train_i)].index.to_list()
                valid_window_i = rec[rec['ind'].isin(valid_i)].index.to_list()
                if set(train_window_i) & set(valid_window_i):
                    raise RuntimeError('train and valid set overlap')
                yield train_window_i, valid_window_i
            else:
                yield train_i, valid_i


def run_benchmark_cv(
        benchmark: Benchmark,  # The Benchmark object
        n_splits: int = 10,  # Number of cross-validations splits
        sample: Union[int, None] = None,  # Sub-sampling factor for debugging.
        group_column: Union[str, None] = None,  #  A column in the data-frame containing grouping information
    ) -> Tuple[pd.DataFrame, pd.DataFrame]: # The results and model predictions 
    
    # Load data and model
    X, y, model, df_subjects = benchmark.load(sample=sample)

    # Create the appropriate CV object
    if group_column is not None:
        if benchmark.features == "raw":
            cv = BdecodeStratifiedByGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = StratifiedByGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        if benchmark.features == "raw":
            cv = BdecodeKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    if group_column is not None:
        if isinstance(X, SliceDataset):
            # May have multiple samples (epochs) per subject -> use rec id
            # to index into df_subjects to get the grouping values
            rec_ids = X.dataset.get_metadata().rec

            # Map rec_ids to subject indices (some subjects may have been removed due to missing
            # data and the indices are thus shifted with respect to the rec_ids)
            rec_to_subject = dict(zip(sorted(rec_ids.unique()), range(len(rec_ids.unique()))))
            subject_indices = [rec_to_subject[r] for r in rec_ids]

            groups = df_subjects[group_column][subject_indices].values
        else:
            groups = df_subjects[group_column].values

        print(f'Found {len(np.unique(groups))} unique groups')
    else:
        groups = None
    
    predict_method = next(method for method in ('predict_proba', 'decision_function', 'predict') if hasattr(model, method))

    print("Running cross validation ...")
    ys_pred = cross_val_predict(model, X, y, cv=cv, groups=groups, method=predict_method)
    print("... done.")

    cv_split_list = list(cv.split(X, y, groups=groups))
    cv_splits = np.concatenate([np.c_[[ii] * len(test), test] for ii, (train, test) in enumerate(cv_split_list)])
    
    # Aggregate across epoch predictions if necessary (i.e. if have more than one prediction per subject)
    if isinstance(X, SliceDataset):
        df = X.dataset.get_metadata()
        if benchmark.task == "classification":
            # Convert the two class-wise probabilities into a single probability
            # and subtract 0.5 for compatibility (to make 0 the decision boundary)
            df['y_pred'] = torch.nn.functional.softmax(torch.tensor(ys_pred), dim=1)[:, 1] - 0.5
        else:
            df['y_pred'] = ys_pred
        df['y_true'] = np.array(y)
        df['cv_split'] = cv_splits[:, 0].astype(int)
        df = df.groupby('rec').mean()

        ys = pd.DataFrame(
            dict(
                y_true=df['y_true'].values,
                y_pred=df['y_pred'].values,
                cv_split=df['cv_split'].astype(int).values,
                participant_id=df_subjects.index
            )
        )
    else:
        ys = pd.DataFrame(dict(y_true=y.flatten(), y_pred=ys_pred.flatten()))
        ys['cv_split'] = 0
        ys.loc[cv_splits[:, 1], 'cv_split'] = cv_splits[:, 0].astype(int)
        ys['participant_id'] = df_subjects.index

    # Compute scores
    target_type = type_of_target(y)
    metrics = {}
    if target_type == 'binary':
        metrics.update({'accuracy': accuracy_score,
                        'accuracy_balanced': balanced_accuracy_score,
                        'AUC': roc_auc_score})
    else:
        metrics.update({'MAE': mean_absolute_error, 'r2_score': r2_score})

    scores = {metric: list() for metric in metrics}
    for split in ys.cv_split.unique():
        for score, scorer in metrics.items():
            y_pred = ys.loc[ys.cv_split == split, "y_pred"].values
            if score in ('accuracy', 'accuracy_balanced'):
                y_pred = (y_pred > 0).astype(int)  # generalize for predict proba
            scores[score].append(
                scorer(ys.loc[ys.cv_split == split, "y_true"].values, y_pred)
            )
    results = pd.DataFrame(scores)        
    for key, value in benchmark.get_params().items():
        results[key] = value
        ys[key] = value

    for metric in scores:
        print(f'{metric}({benchmark}, {benchmark.dataset}) = {results[metric].mean()}')

    ys = ys.set_index('participant_id')
    ys = df_subjects.loc[ys.index].join(ys).reset_index()
    return results, ys
