# coding:utf-8

from .detector import detect
from .stats import quality, IV, VIF, WOE, bin_stats
from .selection import select, stepwise
from .transform import Combiner, WOETransformer
from .metrics import KS, PSI, AUC, KS_bucket, PPSI
from .plot import bin_plot, score_plot, radar_plot
from .scorecard import ScoreCard
# from .widedeepbinary import WideDeepBinary
from .performance import perform, LIFT
from .model import auto_model
from .version import __version__

VERSION = __version__
