import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sma
import statsmodels.formula.api as smf

from .transform import WOETransformer, Combiner
from .utils import to_ndarray, bin_by_splits, save_json, read_json
from .utils.mixin import RulesMixin, BinsMixin


NUMBER_EMPTY = -9999999
NUMBER_INF = 1e10
FACTOR_EMPTY = 'MISSING'
FACTOR_UNKNOWN = 'UNKNOWN'


class ScoreCard(BaseEstimator, RulesMixin, BinsMixin):
    def __init__(self, pdo = 40, rate = 2, base_odds = 1/60, base_score = 600, card = None, combiner = {}, transfer = None, model_type='lr', **kwargs):
        """

        Args:
            card (dict|str|IOBase): dict of card or io to read json
            combiner (toad.Combiner)
            transfer (toad.WOETransformer)
            # self.offset = 363.7244
            # self.factor = 57.7078
            model_type: lr or ols  lr:sklearn.LogisticRegression   ols:statsmodels.ols
        """
        self.pdo = pdo
        self.rate = rate
        self.base_odds = base_odds
        self.base_score = base_score

        self.factor = pdo / np.log(rate)
        self.offset = base_score + self.factor * np.log(base_odds)

        self._combiner = combiner
        self.transfer = transfer
        self.model_type = model_type

        if self.model_type == 'ols':
            self.model = sma
        else:
            self.model = LogisticRegression(**kwargs)

        self._feature_names = None

        if card is not None:
            # self.generate_card(card = card)
            import warnings
            warnings.warn(
                """`ScoreCard(card = {.....})` will be deprecated soon,
                    use `ScoreCard().load({.....})` instead!
                """,
                DeprecationWarning,
            )

            self.load(card)
        
    @property
    def coef_(self):
        """ coef of LR code
        """
        if self.model_type == 'ols':
            return self.model.params[1:]
        else:
            return self.model.coef_[0]
    
    @property
    def intercept_(self):
        if self.model_type == 'ols':
            return self.model.params[0]
        else:
            return self.model.intercept_[0]
    
    @property
    def n_features_(self):
        return (self.coef_ != 0).sum()
    
    @property
    def features_(self):
        if not self._feature_names:
            self._feature_names = list(self.rules.keys())
        
        return self._feature_names
    
    @property
    def combiner(self):
        if not self._combiner:
            # generate a new combiner if not exists
            rules = {}
            for key in self.rules:
                rules[key] = self.rules[key]['bins']
            
            self._combiner = Combiner().load(rules)
        
        return self._combiner


    def fit(self, X, y):
        """
        Args:
            X (2D DataFrame)
            Y (array-like)
        """
        self._feature_names = X.columns.tolist()

        for f in self.features_:
            if f not in self.transfer:
                raise Exception('column \'{f}\' is not in transfer'.format(f = f))

        if self.model_type == 'ols':
            # 增加常数项,截距
            X = sma.add_constant(X)
            self.model = self.model.Logit(y, X).fit()
            print(self.model.summary())
        else:
            self.model.fit(X, y)

        self.rules = self._generate_rules()

        return self

    def predict(self, X, **kwargs):
        """predict score
        Args:
            X (2D array-like): X to predict
            return_sub (bool): if need to return sub score, default `False`

        Returns:
            array-like: predicted score
            DataFrame: sub score for each feature
        """

        if self.model_type == 'ols':
            X = sma.add_constant(X)
            prob = self.model.predict(X)
        else:
            prob = self.model.predict_proba(X)[:, 1]

        result = self.proba_to_score(prob)
        result = np.around(result, decimals=2)
        return result


    def predict_prob(self, X, **kwargs):

        if self.model_type == 'ols':
            X = sma.add_constant(X)
            prob = self.model.predict(X)
        else:
            prob = self.model.predict_proba(X)[:, 1]
        return prob
    

    def _generate_rules(self):
        if not self._check_rules(self.combiner, self.transfer):
            raise Exception('generate failed')
        
        rules = {}
        rules['intercept'] = {
            'bins': np.ndarray([0]),
            'woes': 0,
            'weight': 0,
            'scores': (self.offset - self.factor * self.intercept_,) * 1
        }

        for idx, key in enumerate(self.features_):
            weight = self.coef_[idx]

            if weight == 0:
                continue

            woe = self.transfer[key]['woe']
            
            rules[key] = {
                'bins': self.combiner[key],
                'woes': woe,
                'weight': weight,
                'scores': self.woe_to_score(woe, weight = weight),
            }

        return rules


    def _check_rules(self, combiner, transfer):
        for col in self.features_:
            if col not in combiner:
                raise Exception('column \'{col}\' is not in combiner'.format(col = col))
            
            if col not in transfer:
                raise Exception('column \'{col}\' is not in transfer'.format(col = col))

            l_c = len(combiner[col])
            l_t = len(transfer[col]['woe'])

            if l_c == 0:
                continue

            if np.issubdtype(combiner[col].dtype, np.number):
                if l_c != l_t - 1:
                    raise Exception('column \'{col}\' is not matched, assert {l_t} bins but given {l_c}'.format(col = col, l_t = l_t, l_c = l_c + 1))
            else:
                if l_c != l_t:
                    raise Exception('column \'{col}\' is not matched, assert {l_t} bins but given {l_c}'.format(col = col, l_t = l_t, l_c = l_c))

        return True


    def proba_to_score(self, prob):
        """covert probability to score
        
        odds = (1 - prob) / prob
        score = offset - factor * log(odds)
        """
        return self.factor * (np.log(1 - prob) - np.log(prob)) + self.offset

    def bin_to_score(self, bins, return_sub = False):
        """predict score from bins
        """
        res = bins.copy()
        for col in self.rules:
            if col == 'intercept':
                continue
            s_map = self.rules[col]['scores']
            b = bins[col].values
            # set default group to min score
            b[b == self.EMPTY_BIN] = np.argmin(s_map)
            # replace score
            res[col] = s_map[b]

        score = np.sum(res.values, axis = 1)

        if return_sub is False:
            return score

        return score, res


    def woe_to_score(self, woe, weight = None):
        """calculate score by woe
        """
        woe = to_ndarray(woe)

        if weight is None:
            weight = self.coef_

        b = self.offset - self.factor * self.intercept_
        s = -self.factor * weight * woe

        # drop score whose weight is 0
        mask = 1
        if isinstance(weight, np.ndarray):
            mask = (weight != 0).astype(int)

        # return (s + b / self.n_features_) * mask
        return s * mask

    def _parse_rule(self, rule, **kwargs):
        bins = self.parse_bins(list(rule.keys()))
        scores = np.array(list(rule.values()))

        return {
            'bins': bins,
            'scores': scores,
        }
    
    def _format_rule(self, rule, decimal = 2, **kwargs):
        bins = self.format_bins(rule['bins'])
        scores = np.around(rule['scores'], decimals = decimal).tolist()
        
        return dict(zip(bins, scores))


    def after_export(self, card, to_frame = False, to_json = None, to_csv = None):
        """generate a scorecard object

        Args:
            to_frame (bool): return DataFrame of card
            to_json (str|IOBase): io to write json file
            to_csv (filepath|IOBase): file to write csv

        Returns:
            dict
        """
        if to_json is not None:
            save_json(card, to_json)

        if to_frame or to_csv is not None:
            rows = list()
            for name in card:
                for value, score in card[name].items():
                    rows.append({
                        'name': name,
                        'value': value,
                        'score': score,
                    })

            card = pd.DataFrame(rows)


        if to_csv is not None:
            return card.to_csv(to_csv)

        return card



    def _generate_testing_frame(self, maps, size = 'max', mishap = True, gap = 1e-2):
        """
        Args:
            maps (dict): map of values or splits to generate frame
            size (int|str): size of frame. 'max' (default), 'lcm'
            mishap (bool): is need to add mishap patch to test frame
            gap (float): size of gap for testing border

        Returns:
            DataFrame
        """
        number_patch = np.array([NUMBER_EMPTY, NUMBER_INF])
        factor_patch = np.array([FACTOR_EMPTY, FACTOR_UNKNOWN])

        values = []
        cols = []
        for k, v in maps.items():
            v = np.array(v)
            if np.issubdtype(v.dtype, np.number):
                items = np.concatenate((v, v - gap))
                patch = number_patch
            else:
                # remove else group
                mask = np.argwhere(v == self.ELSE_GROUP)
                if mask.size > 0:
                    v = np.delete(v, mask)

                items = np.concatenate(v)
                patch = factor_patch

            if mishap:
                # add patch to items
                items = np.concatenate((items, patch))

            cols.append(k)
            values.append(np.unique(items))

        # calculate length of values in each columns
        lens = [len(x) for x in values]

        # get size
        if isinstance(size, str):
            if size == 'lcm':
                size = np.lcm.reduce(lens)
            else:
                size = np.max(lens)

        stacks = dict()
        for i in range(len(cols)):
            l = lens[i]
            # generate indexes of value in column
            ix = np.arange(size) % l
            stacks[cols[i]] = values[i][ix]

        return pd.DataFrame(stacks)

    def testing_frame(self, **kwargs):
        """get testing frame with score

        Returns:
            DataFrame: testing frame with score
        """
        maps = self.combiner.export()

        frame = self._generate_testing_frame(maps, **kwargs)
        frame['score'] = self.predict(frame)

        return frame
