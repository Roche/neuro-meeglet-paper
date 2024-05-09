import numpy as np
import pandas as pd
import torch

from braindecode import EEGRegressor, EEGClassifier
from braindecode.models import ShallowFBCSPNet
from braindecode.models.modules import Expression
from braindecode.models.util import to_dense_prediction_model
from braindecode.util import set_random_seeds

from scipy.linalg import eigh, pinv
from skorch.callbacks import LRScheduler
from pyriemann.preprocessing import Whitening

from sklearn.base import BaseEstimator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer, TransformedTargetRegressor
from sklearn.pipeline import make_pipeline

from coffeine.covariance_transformers import (
    Diag,
    LogDiag,
    ExpandFeatures,
    Riemann,
    RiemannSnp,
    NaiveVec
)

from coffeine.spatial_filters import (
    ProjIdentitySpace,
    ProjCommonSpace,
    ProjLWSpace,
    ProjRandomSpace,
    ProjSPoCSpace
)


## Models operating on raw (epoched) data
#

def squeeze_to_ch_x_classes(x):
    """Squeeze the model output from any dimension to batch_size x n_classes."""
    while x.size()[-1] == 1 and x.ndim > 2:
        x = x.squeeze(x.ndim-1)
    return x


def create_raw_model(
    model: str,  # "shallow"
    task: str,  # "regression" or "classification"
    n_channels: int
):  
    """
    Creates an EEGRegressor or EEGClassifier from the ShallowFBCSPNet model
    with cropped decoding and predefined parameters.
    """
    
    # Input Validation
    if model != "shallow":
        raise ValueError("Only the following models are supported: shallow")
    if task not in ["regression", "classification"]:    
        raise ValueError(f"Unsupported task type ({task}).")

    set_random_seeds(seed=20211022, cuda=torch.cuda.is_available())

    model = ShallowFBCSPNet(
        in_chans = n_channels,
        n_classes = 2 if task == "classification" else 1,
        input_window_samples=None,
        final_conv_length=35,
    )
    to_dense_prediction_model(model)  # changes model in place

    if task == "regression":
        # Remove the softmax layer from models
        new_model = torch.nn.Sequential()
        for name, module_ in model.named_children():
            if "softmax" in name:
                continue
            new_model.add_module(name, module_)
        model = new_model

        # Adjustments for cropped decoding
        model.add_module('global_pool', torch.nn.AdaptiveAvgPool1d(1))
        model.add_module('squeeze2', Expression(squeeze_to_ch_x_classes))

        model = EEGRegressor(
            model,
            criterion=torch.nn.L1Loss,
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.000625,
            optimizer__weight_decay=0,
            max_epochs=35,
            train_split=None,  # we do splitting via KFold object in cross_validate
            batch_size=256,
            callbacks=["neg_mean_absolute_error", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=35 - 1))],
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

        model = TransformedTargetRegressor(
            regressor = model,
            transformer=StandardScaler()
        )

    else: # classification
        # Adjustments for cropped decoding
        model.add_module('global_pool', torch.nn.AdaptiveAvgPool1d(1))
        model.add_module('squeeze2', Expression(torch.squeeze))

        model = EEGClassifier(
            model,
            criterion=torch.nn.NLLLoss,
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.000625,
            optimizer__weight_decay=0,
            max_epochs=35,
            train_split=None,  # we do splitting via KFold object in cross_validate
            batch_size=256,
            callbacks=["accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=35 - 1))],
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
    
    return model


## Covariance-based Models
#

def _check_X_df(X):
    if hasattr(X, 'values'):
        X = np.array(list(np.squeeze(X))).astype(float)
        if X.ndim == 2:  # deal with single sample
            assert X.shape[0] == X.shape[1]
            X = X[np.newaxis, :, :]
    return X


def _get_scale(X, scale):
    if scale == 'auto':
        scale = 1 / np.mean([np.trace(x) for x in X])
    return scale


class GroupWhitener(BaseEstimator, TransformerMixin):
    def __init__(self, groups=None, metric='riemann'):
        self.groups = groups
        self.metric = metric
        return self

    def fit(self, X, y):
        X = _check_X_df(X)
        self.whiteners_ = dict()
        for group, inds in X.groupby(self.groups):
            self.whiteners_[group] = Whitening(metric=self.metric).fit(
                X.iloc[inds]
            ) 
        return self      

    def transform(self, X):
        X = _check_X_df(X)
        n_sub, p, p = X.shape
        Xout = np.empty((n_sub, p, p))
        for group, inds in X.groupby(self.groups):
            Xout[inds] = self.whiteners_[group].transform(
                X.iloc[inds]
            )
        return pd.DataFrame({'cov': list(Xout)}) 
        

class Normalizer(BaseEstimator, TransformerMixin):
    def __init__(self, method='trace'):
        self.method=method
    
    def fit(self, X, y=None):
        X = _check_X_df(X)
        return self

    def transform(self, X):
        X = _check_X_df(X)
        n_sub, p, p = X.shape
        Xout = np.empty((n_sub, p, p))
        for sub in range(n_sub):
            C = X[sub]
            Xout[sub] = C / np.trace(C)
        return pd.DataFrame({'cov': list(Xout)}) 


class ReconPCA(BaseEstimator, TransformerMixin):
    def __init__(self, scale='auto', n_compo='full', reg=1e-7):
        self.scale = scale
        self.n_compo = n_compo
        self.reg = reg

    def fit(self, X, y=None):
        X = _check_X_df(X)
        self.n_compo = len(X[0]) if self.n_compo == 'full' else self.n_compo
        self.scale_ = _get_scale(X, self.scale)
        self.filters_ = []
        self.patterns_ = []
        C = X.mean(axis=0)
        eigvals, eigvecs = eigh(C)
        ix = np.argsort(np.abs(eigvals))[::-1]
        evecs = eigvecs[:, ix]
        evecs = evecs[:, :self.n_compo].T
        self.filters_.append(evecs)  # (fb, compo, chan) row vec
        self.patterns_.append(pinv(evecs).T)  # (fb, compo, chan)
        return self

    def transform(self, X):
        X = _check_X_df(X)
        n_sub, p, p = X.shape
        self.n_compo = len(X[0]) if self.n_compo == 'full' else self.n_compo
        Xout = np.empty((n_sub, p, p))
        Xs = self.scale_ * X
        filters = self.filters_[0]  # (compo, chan)
        patterns = self.patterns_[0]
        for sub in range(n_sub):
            C = filters @ Xs[sub] @ filters.T
            C += self.reg * np.eye(self.n_compo)
            Xout[sub] = patterns.T @ C @ patterns
        return pd.DataFrame({'cov': list(Xout)})  # (sub , compo, compo)\


class Whitener(BaseEstimator, TransformerMixin):
    def __init__(self, scale='auto', n_compo='full', metric='riemann'):
        self.scale = scale
        self.n_compo = n_compo
        self.metric = metric
        
    def fit(self, X, y=None):
        X = _check_X_df(X)
        self.n_compo = len(X[0]) if self.n_compo == 'full' else self.n_compo
        self.whitener = Whitening(metric=self.metric, 
                                  dim_red={'n_components': self.n_compo})
        self.whitener.fit(X)
        return self

    def transform(self, X):
        X = _check_X_df(X)
        Xout = self.whitener.transform(X)
        return pd.DataFrame({'cov': list(Xout)})  # (sub , compo, compo)


def make_filter_bank_transformer(
    names, method='riemann',
    projection_params=None,
    vectorization_params=None,
    normalizer=None,
    whitener=None,
    categorical_interaction=None
):
    """
    Generate pipeline for filterbank models.
    
    Prepare filter bank models as used in [1]_. These models take as input
    sensor-space covariance matrices computed from M/EEG signals in different
    frequency bands. Then transformations are applied to improve the
    applicability of linear regression techniques by reducing the impact of
    field spread.

    In terms of implementation, this involves 1) projection
    (e.g. spatial filters) and 2) vectorization (e.g. taking the log on the
    diagonal).
    
    .. note::
        The resulting model expects as inputs data frames in which different
        covarances (e.g. for different frequencies) are stored inside columns
        indexed by ``names``.
        Other columns will be passed through by the underlying column
        transformers.
        The pipeline also supports fitting categorical interaction effects
        after projection and vectorization steps are performed.
    .. note::
        All essential methods from [1]_ are implemented here. In practice,
        we recommend comparing `riemann', `spoc' and `diag' as a baseline.
    
    Parameters
    ----------
    names : list of str
        The column names of the data frame corresponding to different
        covariances.
    method : str
        The method used for extracting features from covariances. Defaults
        to ``'riemann'``. Can be ``'riemann'``, ``'lw_riemann'``, ``'diag'``,
        ``'log_diag'``, ``'random'``, ``'naive'``, ``'spoc'``,
        ``'riemann_wasserstein'``.
    projection_params : dict | None
        The parameters for the projection step.
    vectorization_params : dict | None
        The parameters for the vectorization step.
    whitener: tuple | None
        (method, dict) 
    categorical_interaction : str
        The column in the input data frame containing a binary descriptor
        used to fit 2-way interaction effects.
    
    References
    ----------
    [1] D. Sabbagh, P. Ablin, G. Varoquaux, A. Gramfort, and D.A. Engemann.
        Predictive regression modeling with MEG/EEG: from source power
        to signals and cognitive states.
        *NeuroImage*, page 116893,2020. ISSN 1053-8119.
        https://doi.org/10.1016/j.neuroimage.2020.116893
    """

    # put defaults here for projection and vectorization step
    projection_defaults = {
        'diag': dict(),
        'log_diag': dict(),
        'log_diag_pca': dict(scale=1, n_compo='full', reg=1e-15),
        'random': dict(n_compo='full'),
        'naive': dict(),
        'naive_pca': dict(scale=1, n_compo='full', reg=1e-15),
        'pca_log': dict(scale=1, n_compo='full', reg=1e-15),
        'pca_nolog': dict(scale=1, n_compo='full', reg=1e-15),
        'spoc_log': dict(n_compo='full', scale=1, reg=1e-15, shrink=0.1),
        'spoc_nolog': dict(n_compo='full', scale=1, reg=1e-15, shrink=0.1),
        'riemann': dict(scale=1, n_compo='full', reg=1e-15),
        'lw_riemann': dict(shrink=1),
        'riemann_wasserstein': dict()
    }

    vectorization_defaults = {
        'diag': dict(),
        'log_diag': dict(),
        'log_diag_pca': dict(),
        'random': dict(),
        'naive': dict(method='upper'),
        'naive_pca': dict(method='upper'),
        'pca_log': dict(),
        'pca_nolog': dict(),
        'spoc_log': dict(),
        'spoc_nolog': dict(),
        'riemann': dict(metric='riemann'),
        'riemann_wasserstein': dict(rank='full'),
        'lw_riemann': dict(metric='riemann')
    }

    assert set(projection_defaults) == set(vectorization_defaults)

    if method not in projection_defaults:
        raise ValueError(
            f"The `method` ('{method}') you specified is unknown.")

    # update defaults
    projection_params_ = projection_defaults[method]
    if projection_params is not None:
        projection_params_.update(**projection_params)

    vectorization_params_ = vectorization_defaults[method]
    if vectorization_params is not None:
        vectorization_params_.update(**vectorization_params)

    def _get_projector_vectorizer(projection, vectorization):
        steps = [projection(**projection_params_),
                 vectorization(**vectorization_params_)]
        if whitener is not None:
            steps.insert(0, whitener[0](**whitener[1]))
        if normalizer is not None:
            steps.insert(0, normalizer)
        pipeline = [(make_pipeline(*steps), name) for name in names]
        return pipeline

    # setup pipelines (projection + vectorization step)
    steps = tuple()
    if method == 'pca_log':
        steps = (ProjCommonSpace, LogDiag)
    elif method == 'pca_nolog':
        steps = (ProjCommonSpace, Diag)
    elif method == 'riemann':
        steps = (ProjCommonSpace, Riemann)
    elif method == 'lw_riemann':
        steps = (ProjLWSpace, Riemann)
    elif method == 'diag':
        steps = (ProjIdentitySpace, Diag)
    elif method == 'log_diag':
        steps = (ProjIdentitySpace, LogDiag)
    elif method == 'log_diag_pca':
        steps = (ReconPCA, LogDiag)
    elif method == 'random':
        steps = (ProjRandomSpace, LogDiag)
    elif method == 'naive':
        steps = (ProjIdentitySpace, NaiveVec)
    elif method == 'naive_pca':
        steps = (ReconPCA, NaiveVec)
    elif method == 'spoc_log':
        steps = (ProjSPoCSpace, LogDiag)
    elif method == 'spoc_nolog':
        steps = (ProjSPoCSpace, Diag)
    elif method == 'riemann_wasserstein':
        steps = (ProjIdentitySpace, RiemannSnp)

    filter_bank_transformer = make_column_transformer(
        *_get_projector_vectorizer(*steps), remainder='passthrough')

    if categorical_interaction is not None:
        filter_bank_transformer = ExpandFeatures(
            filter_bank_transformer, expander_column=categorical_interaction)

    return filter_bank_transformer


def create_cov_model(
    model: str,
    task: str,  # "regression" or "classification"
    feature_params: dict,
    feature_info: list = None
):
    if model == 'naive':
        filter_bank_transformer = make_filter_bank_transformer(
            names=feature_info["frequency_names"],
            method='naive'
        )
    elif model in (
            'pca_log', 'pca_nolog', 'riemann', 'spoc_log', 'spoc_nolog',
            'naive_pca', 'log_diag_pca'):
        reg = 0
        if pca_reg := feature_params.get('pca_reg'):
            reg = pca_reg['reg']
        projection_params = feature_params.get(
            "projection_params", dict(scale=1, n_compo=feature_info["data_rank"], reg=reg))
        filter_bank_transformer = make_filter_bank_transformer(
            names=feature_info["frequency_names"],
            method=model,
            projection_params=projection_params,
            whitener = feature_params.get('whitener')
        )
    else:
        raise ValueError('Specified model is not supported!')

    model = make_pipeline(
        filter_bank_transformer, StandardScaler(),
        (RidgeCV if task == 'regression' else RidgeClassifierCV)(alphas=np.logspace(-5, 10, 100))
    )

    return model
