#!/usr/bin/env python
# encoding: utf-8

import os
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from estimators import DataFrameSelector, CombinedAttributesAdder, CombinedLogAttributes
from preprocessing import CategoricalEncoder

PROJECT_DIR = '.'
DATA_DIR = r'F:\Git\data\userprofile'
TRAIN_FILE = 'train_dataset.csv'
TEST_FILE = 'test_dataset.csv'
OUTPUT_PATH = os.path.join(DATA_DIR, "result")

TRAIN_DATA = os.path.join(DATA_DIR, TRAIN_FILE)
TEST_DATA = os.path.join(DATA_DIR, TEST_FILE)


class UserprofileData(object):
    def __init__(self):
        pass

    def __load_data(self, file, is_train):
        columns = ['id', 'is_real_name', 'age', 'is_undegraduate'
            , 'is_blacklist', 'is_4G_ill', 'net_age_m'
            , 'last_pay_m', 'last_pay_amt', 'last_6m_avg_consume', 'm_cost', 'm_balance', 'is_arrears',
                   'charge_sensitivity', 'm_social_persons'
            , 'is_offen_mall', 'last_3m_mavg_mall', 'is_m_WanDa', 'is_m_Sam', 'is_m_movies', 'is_m_tour', 'is_m_pay_gym'
            , 'm_online_shop_app_num', 'm_express_app_num', 'm_finance_app_num', 'm_video_app_num'
            , 'm_airplane_app_num', 'm_train_app_num', 'm_tour_news_app_num'
                   ]
        if is_train:
            columns.append('score')
        data = pd.read_csv(file, header=0, names=columns)
        return data

    def load_train_data(self, file, is_train=1):
        data = self.__load_data(file, is_train)
        y = data["score"]
        X = data.drop("score", axis=1)
        return X, y

    def load_test_data(self, file, is_train=0):
        X = self.__load_data(file, is_train)
        return X


class UserprofilePipeline(object):
    def __init__(self):
        self.num_attribs = ['age', 'net_age_m', 'last_pay_m', 'last_pay_amt', 'last_6m_avg_consume', 'm_cost', 'm_balance'
            , 'charge_sensitivity', 'm_social_persons', 'last_3m_mavg_mall', 'm_online_shop_app_num', 'm_express_app_num'
            , 'm_finance_app_num', 'm_video_app_num', 'm_airplane_app_num', 'm_train_app_num', 'm_tour_news_app_num']
        self.binary_attribs = ['is_real_name', 'is_undegraduate', 'is_blacklist', 'is_4G_ill', 'is_arrears'
            , 'is_offen_mall', 'is_m_WanDa', 'is_m_Sam', 'is_m_movies', 'is_m_tour', 'is_m_pay_gym']
        self.cat_attribs = ['charge_sensitivity']
        self.log_attribs = ['net_age_m', 'm_social_persons', 'm_cost', 'last_6m_avg_consume']
        self.full_pipeline = None

    def __create_pipepine(self):
        ##### Pipeline
        num_pipeline = Pipeline([
            ("selector", DataFrameSelector(self.num_attribs)),
            ("std_scaler", StandardScaler()),
        ])

        binary_pipeline = Pipeline([
            ("selector", DataFrameSelector(self.binary_attribs)),
        ])

        cat_pipeline = Pipeline([
            ("selector", DataFrameSelector(self.cat_attribs)),
            ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
        ])

        extra_pipeline = Pipeline([
            # ("selector", DataFrameSelector(self.log_attribs)),
            ("selector", DataFrameSelector(self.num_attribs)),
            ("extra_feature", CombinedLogAttributes(self.num_attribs)),
            # ("std_scaler", StandardScaler()),
        ])

        self.full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("binary_pipeline", binary_pipeline),
            ("cat_pipeline", cat_pipeline),
            ("extra_pipeline", extra_pipeline),
        ])

    def run_pipeline(self, X):
        self.__create_pipepine()
        return self.full_pipeline.fit_transform(X)


def data_preprocessing(is_train=1):
    userprofile_data = UserprofileData()
    userprofile_pipeline = UserprofilePipeline()
    X, y = None, None
    if is_train:
        X, y = userprofile_data.load_train_data(TRAIN_DATA)
    else:
        X = userprofile_data.load_test_data(TEST_DATA)
    X_prepared = userprofile_pipeline.run_pipeline(X)
    return X_prepared, y


if __name__ == "__main__":
    # userprofile_pipeline = UserprofilePipeline()
    # print(dir(userprofile_pipeline))
    X_prepared, y = data_preprocessing()
    print(X_prepared.shape)
    print(X_prepared[0])