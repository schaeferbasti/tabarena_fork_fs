from __future__ import annotations

from math import log
import numpy as np
import pandas as pd

import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class mRMR:
    """mRMR feature selector"""

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

        feature_ranking = self.mrmr(X_np, y_np, n_selected_features=n_max_features)

        selected_features_idx = feature_ranking[:n_max_features]
        selected_features = X.columns[selected_features_idx]

        X_selected = X[selected_features]
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]

    def mrmr(self, X, y, **kwargs):
        """
        This function implements the MRMR feature selection

        Input
        -----
        X: {numpy array}, shape (n_samples, n_features)
            input data, guaranteed to be discrete
        y: {numpy array}, shape (n_samples,)
            input class labels
        kwargs: {dictionary}
            n_selected_features: {int}
                number of features to select

        Output
        ------
        F: {numpy array}, shape (n_features,)
            index of selected features, F[0] is the most important feature
        J_CMI: {numpy array}, shape: (n_features,)
            corresponding objective function value of selected features
        MIfy: {numpy array}, shape: (n_features,)
            corresponding mutual information between selected features and response

        Reference
        ---------
        Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
        """
        if 'n_selected_features' in kwargs.keys():
            n_selected_features = kwargs['n_selected_features']
            F, J_CMI, MIfy = self.lcsi(X, y, gamma=0, function_name='MRMR', n_selected_features=n_selected_features)
        else:
            F, J_CMI, MIfy = self.lcsi(X, y, gamma=0, function_name='MRMR')
        return F

    def lcsi(self, X, y, **kwargs):
        """
        This function implements the basic scoring criteria for linear combination of shannon information term.
        The scoring criteria is calculated based on the formula j_cmi=I(f;y)-beta*sum_j(I(fj;f))+gamma*sum(I(fj;f|y))

        Input
        -----
        X: {numpy array}, shape (n_samples, n_features)
            input data, guaranteed to be a discrete data matrix
        y: {numpy array}, shape (n_samples,)
            input class labels
        kwargs: {dictionary}
            Parameters for different feature selection algorithms.
            beta: {float}
                beta is the parameter in j_cmi=I(f;y)-beta*sum(I(fj;f))+gamma*sum(I(fj;f|y))
            gamma: {float}
                gamma is the parameter in j_cmi=I(f;y)-beta*sum(I(fj;f))+gamma*sum(I(fj;f|y))
            function_name: {string}
                name of the feature selection function
            n_selected_features: {int}
                number of features to select

        Output
        ------
        F: {numpy array}, shape: (n_features,)
            index of selected features, F[0] is the most important feature
        J_CMI: {numpy array}, shape: (n_features,)
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
        J_CMI = []
        # Mutual information between feature and response
        MIfy = []
        # indicate whether the user specifies the number of features
        is_n_selected_features_specified = False
        # initialize the parameters
        if 'beta' in kwargs.keys():
            beta = kwargs['beta']
        if 'gamma' in kwargs.keys():
            gamma = kwargs['gamma']
        if 'n_selected_features' in kwargs.keys():
            n_selected_features = kwargs['n_selected_features']
            is_n_selected_features_specified = True

        # select the feature whose j_cmi is the largest
        # t1 stores I(f;y) for each feature f
        t1 = np.zeros(n_features)
        # t2 stores sum_j(I(fj;f)) for each feature f
        t2 = np.zeros(n_features)
        # t3 stores sum_j(I(fj;f|y)) for each feature f
        t3 = np.zeros(n_features)
        for i in range(n_features):
            f = X[:, i]
            t1[i] = self.midd(f, y)

        # make sure that j_cmi is positive at the very beginning
        j_cmi = 1

        while True:
            if len(F) == 0:
                # select the feature whose mutual information is the largest
                idx = np.argmax(t1)
                F.append(idx)
                J_CMI.append(t1[idx])
                MIfy.append(t1[idx])
                f_select = X[:, idx]

            if is_n_selected_features_specified:
                if len(F) == n_selected_features:
                    break
            else:
                if j_cmi < 0:
                    break

            # we assign an extreme small value to j_cmi to ensure it is smaller than all possible values of j_cmi
            j_cmi = -1E30
            if 'function_name' in kwargs.keys():
                if kwargs['function_name'] == 'MRMR':
                    beta = 1.0 / len(F)
                elif kwargs['function_name'] == 'JMI':
                    beta = 1.0 / len(F)
                    gamma = 1.0 / len(F)
            for i in range(n_features):
                if i not in F:
                    f = X[:, i]
                    t2[i] += self.midd(f_select, f)
                    t3[i] += self.cmidd(f_select, f, y)
                    # calculate j_cmi for feature i (not in F)
                    t = t1[i] - beta * t2[i] + gamma * t3[i]
                    # record the largest j_cmi and the corresponding feature index
                    if t > j_cmi:
                        j_cmi = t
                        idx = i
            F.append(idx)
            J_CMI.append(j_cmi)
            MIfy.append(t1[idx])
            f_select = X[:, idx]

        return np.array(F), np.array(J_CMI), np.array(MIfy)


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
