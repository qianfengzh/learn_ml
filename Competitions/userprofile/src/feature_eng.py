#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from estimators import DataFrameSelector, CombinedAttributesAdder, CombinedLogAttributes, CombinedBiSumAttributes
from preprocessing import CategoricalEncoder

# PROJECT_DIR = '.'
# DATA_DIR = r'F:\Git\data\userprofile'
# TRAIN_FILE = 'train_dataset.csv'
# TEST_FILE = 'test_dataset.csv'
# OUTPUT_PATH = os.path.join(DATA_DIR, "result")
#
# TRAIN_DATA = os.path.join(DATA_DIR, TRAIN_FILE)
# TEST_DATA = os.path.join(DATA_DIR, TEST_FILE)


class UserprofileData(object):
    def __init__(self, name=None, is_train=None, file=None):
        self.name = name
        self.is_train = is_train
        self.id = None
        self.data = None
        self.label = None
        self.__load_data(file, self.is_train)

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
        if is_train:
            self.label = data["score"]
            self.data = data.drop(["id", "score"], axis=1)
        else:
            self.data = data.drop("id", axis=1)
        self.id = data["id"]
        return self

    # def load_train_data(self, file, is_train=1):
    #     data = self.__load_data(file, is_train)
    #     y = data["score"]
    #     X = data.drop("score", axis=1)
    #     return X, y
    #
    # def load_test_data(self, file, is_train=0):
    #     X = self.__load_data(file, is_train)
    #     return X


class UserprofilePipeline(object):
    def __init__(self):
        self.num_attribs = ['age', 'net_age_m', 'last_pay_m', 'last_pay_amt', 'last_6m_avg_consume', 'm_cost', 'm_balance'
            , 'charge_sensitivity', 'm_social_persons', 'last_3m_mavg_mall', 'm_online_shop_app_num', 'm_express_app_num'
            , 'm_finance_app_num', 'm_video_app_num', 'm_airplane_app_num', 'm_train_app_num', 'm_tour_news_app_num']
        self.binary_attribs = ['is_real_name', 'is_undegraduate', 'is_blacklist', 'is_4G_ill', 'is_arrears'
            , 'is_offen_mall', 'is_m_WanDa', 'is_m_Sam', 'is_m_movies', 'is_m_tour', 'is_m_pay_gym']
        self.cat_attribs = ['charge_sensitivity']
        # self.log_attribs = ['net_age_m', 'm_social_persons', 'm_cost', 'last_6m_avg_consume']
        # self.bi_sum_attribs = ['is_4G_ill','is_arrears','is_m_tour','is_m_WanDa','is_offen_mall','is_m_pay_gym']
        self.full_pipeline = None

    def __create_pipepine(self):
        ##### Pipeline
        num_pipeline = Pipeline([
            ("selector", DataFrameSelector(self.num_attribs)),
            # ("std_scaler", StandardScaler()),
        ])

        binary_pipeline = Pipeline([
            ("selector", DataFrameSelector(self.binary_attribs)),
        ])

        cat_pipeline = Pipeline([
            ("selector", DataFrameSelector(self.cat_attribs)),
            ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
        ])

        log_pipeline = Pipeline([
            # ("selector", DataFrameSelector(self.log_attribs)),
            ("selector", DataFrameSelector(self.num_attribs)),
            ("log_feature", CombinedLogAttributes(self.num_attribs)),
            # ("std_scaler", StandardScaler()),
        ])

        # bi_sum_pipeline = Pipeline([
        #     ("selector", DataFrameSelector(self.bi_sum_attribs)),
        #     ("bi_sum_feature", CombinedBiSumAttributes(self.bi_sum_attribs)),
        # ])

        self.full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("binary_pipeline", binary_pipeline),
            ("cat_pipeline", cat_pipeline),
            ("log_pipeline", log_pipeline),
            # ("bi_sum_pipeline", bi_sum_pipeline),
        ])

    def run_pipeline(self, X):
        self.__create_pipepine()
        return self.full_pipeline.fit_transform(X)


def __outlier_filter(X, y=None, attribs=None):
    idx_list = []
    stats = X[attribs].describe()
    IQR = stats.loc["75%"] - stats.loc["25%"]
    c_max = stats.loc["75%"] + 1.5 * IQR
    c_min = stats.loc["25%"] - 1.5 * IQR
    for attrib in attribs:
        idxs = X[(X[attrib] >= c_max.loc[attrib]) | (X[attrib] <= c_min.loc[attrib])].index.values
        # print(attrib)
        # print(len(idxs))
        idx_list.extend(idxs)
    idx_set = set(idx_list)
    print("[outlier] filter rows: ", len(idx_set))
    X.drop(idx_set, inplace=True)
    X.reset_index(drop=True, inplace=True)
    if y is not None:
        y.drop(idx_set, inplace=True)
        y.reset_index(drop=True, inplace=True)
    # y = np.delete(y, list(idx_set),axis=0)
    return X, y


def __feature_less_than_filter(X, y=None, **attribs):
    """
    :param X:
    :param y:
    :param attribs:
    :return:
        X: DataFrame
        y: ndarray
    """
    idx_list = []
    for feature, min_value in attribs.items():
        idxs = X[X[feature] < min_value].index.values
        idx_list.extend(idxs)
    idx_set = set(idx_list)
    print("["+ feature +"<"+ str(min_value) +"] filter rows: ", len(idx_set))
    X.drop(idx_set, inplace=True)
    X.reset_index(drop=True, inplace=True)
    if y is not None:
        y.drop(idx_set, inplace=True)
        y.reset_index(drop=True, inplace=True)
    return X, y



def data_preprocessing(userprofile_data, is_train=1):
    # userprofile_data = UserprofileData()
    userprofile_pipeline = UserprofilePipeline()
    X, y = None, None
    if is_train:
        X, y = userprofile_data.data, userprofile_data.label
        # # 训练异常数据清洗
        # filter_attribs_less_than = {"age": 18}
        # X, y = __feature_less_than_filter(X, y, **filter_attribs_less_than)
    else:
        X = userprofile_data.data

    # Pipe 进行数据处理
    X = userprofile_pipeline.run_pipeline(X)

    # 数据列补充
    columns = []
    columns.extend(userprofile_pipeline.num_attribs)
    columns.extend(userprofile_pipeline.binary_attribs)

    cat_attribs = []
    for i in range(6):
        cat_attribs.append(userprofile_pipeline.cat_attribs[0] + "_" + str(i))
    columns.extend(cat_attribs)

    log_attribs = "*log_".join(userprofile_pipeline.num_attribs).split("*")
    log_attribs[0] = "log_" + log_attribs[0]
    columns.extend(log_attribs)

    # columns.append("bi_sum")
    X = pd.DataFrame(X, columns=columns)

    if is_train:
        # 训练数据清洗，对轻微异常分布的数据删除
        # filter_attribs = ["log_m_social_persons", "log_last_6m_avg_consume", "log_m_cost"]
        # X, y = __outlier_filter(X, y, filter_attribs)
        # return X.values, y.values.reshape((-1, 1))
        return X, y.values.reshape((-1, 1))
    else:
        return X.values
    # return X_prepared_filtered, y_filtered


if __name__ == "__main__":
    # userprofile_pipeline = UserprofilePipeline()
    # print(dir(userprofile_pipeline))

    X, y = data_preprocessing()
    print("X_train shape: ", X.shape)
    print(X[0])
    print("y_train shape: ", y.shape)

    X = data_preprocessing(is_train=0)
    print("X_test shape: ", X.shape)
    print(X[0])
