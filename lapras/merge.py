# coding:utf-8

import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.cluster import KMeans
from .utils import fillna, bin_by_splits, to_ndarray, clip
from .utils.decorator import support_dataframe
from .utils.forwardSplit import *

DEFAULT_BINS = 10
DEFAULT_DECIMAL = 4 # 默认小数精度


def MonoMerge(feature, target, n_bins=None, min_samples=10):
    '''
    :param feature: 待分箱变量
    :param target: 目标特征
    :param n_bins: 最多分箱数
    :param min_sample: 每个分箱最小样本数
    :return: numpy array -- 切割点数组
    '''
    df = pd.DataFrame()
    df['feature'] = feature
    df['target'] = target
    df['feature'] = df['feature'].fillna(-99) #有缺失值则无法分箱

    if n_bins is None:
        n_bins = DEFAULT_BINS

    if min_samples < 1:
        min_samples = math.ceil(len(target) * min_samples)

    t = forwardSplit(df['feature'], df['target'])
    t.fit(sby='woe', minv=0.01, num_split=n_bins-1,min_sample=min_samples,init_split=20)

    # 单调分箱失败，采用决策树分2箱
    if t.bins is None:
        bins = DTMerge(feature, target, n_bins=2, min_samples=min_samples)
        return bins

    else:
        bins = list(t.bins)
        bins.pop(0) # 删除分箱两边的边界值
        bins.pop(-1) # 删除分箱两边的边界值
        bins = list(set(bins))

        # 结果取4位小数
        thresholds = np.array(bins)
        for i in range(len(thresholds)):
            if type(thresholds[i]) == np.float64:
                thresholds[i] = round(thresholds[i], DEFAULT_DECIMAL)

        return np.sort(thresholds)


def DTMerge(feature, target, nan = -1, n_bins = None, min_samples = 1):
    """Merge by Decision Tree

    Args:
        feature (array-like)
        target (array-like): target will be used to fit decision tree
        nan (number): value will be used to fill nan
        n_bins (int): n groups that will be merged into
        min_samples (int): min number of samples in each leaf nodes

    Returns:
        array: array of split points
    """
    if n_bins is None and min_samples == 1:
        n_bins = DEFAULT_BINS

    feature = fillna(feature, by = nan)

    tree = DecisionTreeClassifier(
        min_samples_leaf = min_samples,
        max_leaf_nodes = n_bins,
    )
    tree.fit(feature.reshape((-1, 1)), target)

    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != _tree.TREE_UNDEFINED]

    # 结果取4位小数
    for i in range(len(thresholds)):
        if type(thresholds[i]) == np.float64:
            thresholds[i] = round(thresholds[i],DEFAULT_DECIMAL)

    return np.sort(thresholds)


def StepMerge(feature, nan = None, n_bins = None, clip_v = None, clip_std = None, clip_q = None,min_samples = 1):
    """Merge by step

    Args:
        feature (array-like)
        nan (number): value will be used to fill nan
        n_bins (int): n groups that will be merged into
        clip_v (number | tuple): min/max value of clipping
        clip_std (number | tuple): min/max std of clipping
        clip_q (number | tuple): min/max quantile of clipping
    Returns:
        array: split points of feature
    """
    if n_bins is None:
        n_bins = DEFAULT_BINS

    if nan is not None:
        feature = fillna(feature, by = nan)

    feature = clip(feature, value = clip_v, std = clip_std, quantile = clip_q)

    max = np.nanmax(feature)
    min = np.nanmin(feature)

    step = (max - min) / n_bins
    return np.arange(min, max, step)[1:].round(4)


def QuantileMerge(feature, nan = -1, n_bins = None, q = None ,min_samples = 1):
    """Merge by quantile

    Args:
        feature (array-like)
        nan (number): value will be used to fill nan
        n_bins (int): n groups that will be merged into
        q (array-like): list of percentage split points

    Returns:
        array: split points of feature
    """
    if n_bins is None and q is None:
        n_bins = DEFAULT_BINS

    if q is None:
        step = 1 / n_bins
        q = np.arange(0, 1, step)

    feature = fillna(feature, by = nan)

    splits = np.quantile(feature, q).round(4)

    return np.unique(splits)[1:]


def KMeansMerge(feature, target = None, nan = -1, n_bins = None, random_state = 1, min_samples = 1):
    """Merge by KMeans

    Args:
        feature (array-like)
        target (array-like): target will be used to fit kmeans model
        nan (number): value will be used to fill nan
        n_bins (int): n groups that will be merged into
        random_state (int): random state will be used for kmeans model

    Returns:
        array: split points of feature
    """
    if n_bins is None:
        n_bins = DEFAULT_BINS

    feature = fillna(feature, by = nan)

    model = KMeans(
        n_clusters = n_bins,
        random_state = random_state
    )
    model.fit(feature.reshape((-1 ,1)), target)

    centers = np.sort(model.cluster_centers_.reshape(-1))

    l = len(centers) - 1
    splits = np.zeros(l)
    for i in range(l):
        splits[i] = (centers[i] + centers[i+1]) / 2
    return splits.round(4)


@support_dataframe(require_target = False)
def merge(feature, target = None, method = 'dt', return_splits = False, **kwargs):
    """merge feature into groups

    Args:
        feature (array-like)
        target (array-like)
        method (str): 'dt', 'chi', 'quantile', 'step', 'kmeans' - the strategy to be used to merge feature
        return_splits (bool): if needs to return splits
        n_bins (int): n groups that will be merged into
    Returns:
        array: a array of merged label with the same size of feature
        array: list of split points
    """
    feature = to_ndarray(feature)
    method = method.lower()

    if method == 'dt':
        splits = DTMerge(feature, target, **kwargs)
    elif method == 'mono':
        splits = MonoMerge(feature, target, **kwargs)
    elif method == 'quantile':
        splits = QuantileMerge(feature, **kwargs)
    elif method == 'step':
        splits = StepMerge(feature, **kwargs)
    elif method == 'kmeans':
        splits = KMeansMerge(feature, target=target, **kwargs)
    else:
        splits = np.empty(shape=(0,))

    if len(splits):
        bins = bin_by_splits(feature, splits)
    else:
        bins = np.zeros(len(feature))

    if return_splits:
        return bins, splits
    # print(splits)
    return bins


