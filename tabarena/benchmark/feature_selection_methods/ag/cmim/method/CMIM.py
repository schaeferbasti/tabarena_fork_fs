from __future__ import annotations

from math import log
import numpy as np
import pandas as pd

import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class CMIM:
    """CMIM feature selector"""

    def __init__(self):
        self._y = None
        self._model = None
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        feature_ranking = self.cmim(X_np, y_np, n_selected_features=n_max_features)

        selected_features_idx = feature_ranking[:n_max_features]
        selected_features = X.columns[selected_features_idx]

        X_selected = X[selected_features]
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]

    def cmim(self, X, y, **kwargs):
        """
        This function implements the CMIM feature selection.
        The scoring criteria is calculated based on the formula j_cmim=I(f;y)-max_j(I(fj;f)-I(fj;f|y))

        Input
        -----
        X: {numpy array}, shape (n_samples, n_features)
            Input data, guaranteed to be a discrete numpy array
        y: {numpy array}, shape (n_samples,)
            guaranteed to be a numpy array
        kwargs: {dictionary}
            n_selected_features: {int}
                number of features to select

        Output
        ------
        F: {numpy array}, shape (n_features,)
            index of selected features, F[0] is the most important feature
        J_CMIM: {numpy array}, shape: (n_features,)
            corresponding objective function value of selected features
        MIfy: {numpy array}, shape: (n_features,)
            corresponding mutual information between selected features and response

        Reference
        ---------
        Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
        """

        n_samples, n_features = X.shape
        # index of selected features, initialized to be empty
        F = []
        # Objective function value for selected features
        J_CMIM = []
        # Mutual information between feature and response
        MIfy = []
        # indicate whether the user specifies the number of features
        is_n_selected_features_specified = False

        if 'n_selected_features' in kwargs.keys():
            n_selected_features = kwargs['n_selected_features']
            is_n_selected_features_specified = True

        # t1 stores I(f;y) for each feature f
        t1 = np.zeros(n_features)

        # max stores max(I(fj;f)-I(fj;f|y)) for each feature f
        # we assign an extreme small value to max[i] ito make it is smaller than possible value of max(I(fj;f)-I(fj;f|y))
        max = -10000000 * np.ones(n_features)
        for i in range(n_features):
            f = X[:, i]
            t1[i] = self.midd(f, y)

        # make sure that j_cmi is positive at the very beginning
        j_cmim = 1

        while True:
            if len(F) == 0:
                # select the feature whose mutual information is the largest
                idx = np.argmax(t1)
                F.append(idx)
                J_CMIM.append(t1[idx])
                MIfy.append(t1[idx])
                f_select = X[:, idx]

            if is_n_selected_features_specified:
                if len(F) == n_selected_features:
                    break
            else:
                if j_cmim <= 0:
                    break

            # we assign an extreme small value to j_cmim to ensure it is smaller than all possible values of j_cmim
            j_cmim = -1000000000000
            for i in range(n_features):
                if i not in F:
                    f = X[:, i]
                    t2 = self.midd(f_select, f)
                    t3 = self.cmidd(f_select, f, y)
                    if t2 - t3 > max[i]:
                        max[i] = t2 - t3
                    # calculate j_cmim for feature i (not in F)
                    t = t1[i] - max[i]
                    # record the largest j_cmim and the corresponding feature index
                    if t > j_cmim:
                        j_cmim = t
                        idx = i
            F.append(idx)
            J_CMIM.append(j_cmim)
            MIfy.append(t1[idx])
            f_select = X[:, idx]
        return np.array(F)

    def conditional_entropy(self, f1, f2):
        """
        This function calculates the conditional entropy, where ce = H(f1) - I(f1;f2)

        Input
        -----
        f1: {numpy array}, shape (n_samples,)
        f2: {numpy array}, shape (n_samples,)

        Output
        ------
        ce: {float}
            ce is conditional entropy of f1 and f2
        """

        ce = self.entropyd(f1) - self.midd(f1, f2)
        return ce

    def midd(self, x, y):
        """
        Discrete mutual information estimator given a list of samples which can be any hashable object
        """

        return -self.entropyd(list(zip(x, y))) + self.entropyd(x) + self.entropyd(y)

    def entropyd(self, sx, base=2):
        """
        Discrete entropy estimator given a list of samples which can be any hashable object
        """

        return self.entropyfromprobs(self.hist(sx), base=base)

    def cmidd(self, x, y, z):
        """
        Discrete mutual information estimator given a list of samples which can be any hashable object
        """

        return self.entropyd(list(zip(y, z))) + self.entropyd(list(zip(x, z))) - self.entropyd(
            list(zip(x, y, z))) - self.entropyd(z)

    def hist(self, sx):
        # Histogram from list of samples
        d = dict()
        for s in sx:
            d[s] = d.get(s, 0) + 1
        return map(lambda z: float(z) / len(sx), d.values())

    def entropyfromprobs(self, probs, base=2):
        # Turn a normalized list of probabilities of discrete outcomes into entropy (base 2)
        return -sum(map(self.elog, probs)) / log(base)

    def elog(self, x):
        # for entropy, 0 log 0 = 0. but we get an error for putting log 0
        if x <= 0. or x >= 1.:
            return 0
        else:
            return x * log(x)