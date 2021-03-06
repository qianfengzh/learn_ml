#!/usr/bin/env python
# encoding: utf-8


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

# DataFrameSelector
from sklearn.utils import check_array


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.__attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.__attribute_names]


class CombinedLogAttributes(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = "*log_".join(attribute_names).split("*")
        self.attribute_names[0] = "log_" + self.attribute_names[0]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        log_df = X.apply(lambda x: np.log(x+5))
        return log_df.values


class CombinedBiSumAttributes(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        log_df = X.sum(axis=1)
        return log_df.values


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, df_data):
        data_columns = df_data.columns
        self.ix_net_age_m = data_columns.get_loc("net_age_m")
        self.ix_age = data_columns.get_loc("age")
        self.ix_last_6m_avg_consume = data_columns.get_loc("last_6m_avg_consume")
        self.ix_m_social_persons = data_columns.get_loc("m_social_persons")
        self.ix_m_cost = data_columns.get_loc("m_cost")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(self.ix_net_age_m)
        # m_net_age_per_age = X[:, self.ix_net_age_m] / (12*X[:, self.ix_age])
        last_6m_consum_per_net_age = X[:, self.ix_last_6m_avg_consume] / X[:, self.ix_net_age_m]
        print(X[:, self.ix_net_age_m] + 1)
        net_age_m_log = np.log10(np.asarray((X[:, self.ix_net_age_m] + 1), dtype=float))
        m_social_persons_log = np.log10(np.asarray(X[:, self.ix_m_social_persons] + 1, dtype=float))
        m_cost_log = np.log10(np.asarray(X[:, self.ix_m_cost] + 1, dtype=float))
        last_6m_avg_consume_log = np.log10(np.asarray(X[:, self.ix_last_6m_avg_consume] + 1, dtype=float))
        return np.c_[X, last_6m_consum_per_net_age, net_age_m_log
            , m_social_persons_log, m_cost_log, last_6m_avg_consume_log]


class StackingEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))
        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))
        return X_transformed

# ############################
# # Train the stacked models #
# ############################
#
# stacked_pipeline = make_pipeline(
#     StackingEstimator(estimator=LassoLarsCV(normalize=True)),
#     StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001,
#                                                           loss="huber",
#                                                           max_depth=3,
#                                                           max_features=0.55,
#                                                           min_samples_leaf=18,
#                                                           min_samples_split=14,
#                                                           subsample=0.7),
#                      ),
#     LassoLarsCV()
# )
#
# stacked_pipeline.fit(finaltrainset, y_train)
#
# #########################
# # Predict the test data #
# #########################
#
# results = stacked_pipeline.predict(finaltestset)
