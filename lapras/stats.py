from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from scipy import stats
from .merge import merge

from .utils import (
    np_count,
    np_unique,
    to_ndarray,
    feature_splits,
    is_continuous,
    inter_feature,
    split_target,
)

from .utils.decorator import support_dataframe

STATS_EMPTY = np.nan


def probability(target, mask = None):
    """get probability of target by mask
    """
    if mask is None:
        return 1, 1

    counts_0 = np_count(target, 0, default = 1)
    counts_1 = np_count(target, 1, default = 1)

    sub_target = target[mask]

    sub_0 = np_count(sub_target, 0, default = 1)
    sub_1 = np_count(sub_target, 1, default = 1)

    y_prob = sub_1 / counts_1
    n_prob = sub_0 / counts_0

    return y_prob, n_prob


def WOE(y_prob, n_prob):
    """get WOE of a group

    Args:
        y_prob: the probability of grouped y in total y
        n_prob: the probability of grouped n in total n

    Returns:
        number: woe value
    """
    return np.log(y_prob / n_prob)


def _IV(feature, target):
    """private information value func

    Args:
        feature (array-like)
        target (array-like)

    Returns:
        number: IV
    """
    feature = to_ndarray(feature)
    target = to_ndarray(target)

    value = 0

    for v in np.unique(feature):
        y_prob, n_prob = probability(target, mask = (feature == v))

        value += (y_prob - n_prob) * WOE(y_prob, n_prob)

    return value


@support_dataframe
def IV(feature, target, **kwargs):
    """get the IV of a feature

    Args:
        feature (array-like)
        target (array-like)
        n_bins (int): n groups that the feature will bin into
        method (str): the strategy to be used to merge feature, default is 'dt'
        **kwargs (): other options for merge function
    """
    if not is_continuous(feature):
        return _IV(feature, target)

    feature = merge(feature, target, **kwargs)

    return _IV(feature, target)


def badrate(target):
    """calculate badrate

    Args:
        target (array-like): target array which `1` is bad

    Returns:
        float
    """
    return np.sum(target) / len(target)


def VIF(frame):
    """calculate vif

    Args:
        frame (ndarray|DataFrame)

    Returns:
        Series
    """
    frame = frame.fillna(-99)
    index = None
    if isinstance(frame, pd.DataFrame):
        index = frame.columns
        frame = frame.values
    
    from sklearn.linear_model import LinearRegression

    model = LinearRegression(fit_intercept = False)

    l = frame.shape[1]
    vif = np.zeros(l)

    for i in range(l):
        X = frame[:, np.arange(l) != i]
        y = frame[:, i]
        model.fit(X, y)

        pre_y = model.predict(X)

        vif[i] = np.sum((y - np.mean(y)) ** 2) / np.sum((pre_y - y) ** 2)
    
    return pd.Series(vif, index = index)


def column_quality(feature, target, name = 'feature', **kwargs):
    """calculate quality of a feature

    Args:
        feature (array-like)
        target (array-like)
        name (str): feature's name that will be setted in the returned Series

    Returns:
        Series: a list of quality with the feature's name
    """
    feature = to_ndarray(feature)
    target = to_ndarray(target)

    if not np.issubdtype(feature.dtype, np.number):
        feature = feature.astype(str)

    c = len(np_unique(feature))
    iv = STATS_EMPTY

    # skip when unique is too much
    if is_continuous(feature) or c / len(feature) < 0.5:
        iv = IV(feature, target, **kwargs)

    row = pd.Series(
        index = ['iv', 'unique'],
        data = [iv, c],
    )

    row.name = name
    return row


def quality(dataframe, target = 'target',**kwargs):
    """get quality of features in data

    Args:
        dataframe (DataFrame): dataframe that will be calculate quality
        target (str): the target's name in dataframe

    Returns:
        DataFrame: quality of features with the features' name as row name
    """
    frame, target = split_target(dataframe, target)
    
    res = []
    for name, series in frame.iteritems():
        r = column_quality(series, target,name=name)
        res.append(r)

    result = pd.DataFrame(res).sort_values(
        by = 'iv',
        ascending = False,
    )
    result['iv'] = result['iv'].round(decimals=6)
    return result


def bin_stats(frame, col=None, target='target'):
    """return detailed inforrmatiom for bins

    Args:
        frame (DataFrame)
        x (str): column in frame that will be used as x axis
        target (str): target column in frame

    """
    frame = frame.copy()
    group = frame.groupby(col)
    table = group[target].agg(['sum', 'count']).reset_index()
    table.columns = [col,'bad_count','total_count']
    table['bad_rate'] = table['bad_count'] / table['total_count']
    table['ratio'] = table['total_count'] / table['total_count'].sum()

    X = to_ndarray(frame[col])
    value = np.unique(X)
    l = len(value)
    woe = np.zeros(l)
    iv = np.zeros(l)
    for i in range(l):
        y_prob, n_prob = probability(frame[target], mask=(X == value[i]))
        woe[i] = np.log(y_prob / n_prob)
        iv[i] = woe[i] * (y_prob - n_prob)
    table['woe'] = woe
    table['iv'] = iv
    table['total_iv'] = table.iv.replace({np.inf: 0, -np.inf: 0}).sum()

    return table

