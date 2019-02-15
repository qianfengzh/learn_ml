#!/usr/bin/env python
# encoding: utf-8
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from feature_eng import data_preprocessing, UserprofileData
from metrics import submit_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


PROJECT_DIR = '.'
DATA_DIR = r'F:\Git\data\userprofile'
TRAIN_FILE = 'train_dataset.csv'
TEST_FILE = 'test_dataset.csv'
OUTPUT_PATH = os.path.join(DATA_DIR, "result")

TRAIN_DATA = os.path.join(DATA_DIR, TRAIN_FILE)
TEST_DATA = os.path.join(DATA_DIR, TEST_FILE)

FILE_NAME = "lin_reg_base_line-v3.csv"


################## 模型测试 ###############
lin_reg = LinearRegression()
# featureV1: 常规处理，未增加特征
# train_score: 0.0455
#------------------------------
# featureV2: 常规处理，增加4个log处理特征（['net_age_m', 'm_social_persons', 'm_cost', 'last_6m_avg_consume']）
# train_score: 0.057
#------------------------------
# featureV3: 常规处理，对所有数字类特征增加log处理特征
# train_score: 0.058
#------------------------------
# featureV4: 常规处理，对所有数字类特征增加log处理特征，对 age<18 的数据进行过滤清洗
# train_score: 0.059
#------------------------------
# featureV5: 常规处理，对所有数字类特征增加log处理特征，对 age<18 的数据进行过滤清洗, 3 个log特征 outlier 清理
# （["log_m_social_persons", "log_last_6m_avg_consume", "log_m_cost"]）
# train_score: 0.060
#------------------------------

rf_reg = RandomForestRegressor(n_estimators=10, max_depth=5, criterion="mae", max_features="sqrt")
# featureV1: 常规处理，未增加特征
# n_estimators=10, max_depth=5, criterion="mae", max_features="sqrt"
# train_score: 0.051  test_score: 0.049

gbr_reg = GradientBoostingRegressor(loss='huber', learning_rate=0.25, n_estimators=80
                                    , subsample=0.80, max_depth=5, max_features="sqrt")

def cross_val(X_train, y_train, model=lin_reg):
    X, y = X_train, y_train
    # print("X shape: ", X.shape)
    # print("y shape: ", y.shape)
    rs = ShuffleSplit(n_splits=5, test_size=.2, random_state=142)
    scores_train = []
    scores_test = []
    for train_idx, test_idx in rs.split(X):
        train_X = X[train_idx]
        train_y = y[train_idx]
        test_X = X[test_idx]
        test_y = y[test_idx]
        model.fit(train_X, train_y)
        scores_train.append(submit_score(train_y, model.predict(train_X)))
        scores_test.append(submit_score(test_y, model.predict(test_X)))
        print("=====================")
        print("train score: ", scores_train[-1])
        print("test score: ", scores_test[-1])
    print("mean-train score: ", np.array(scores_train).mean())
    print("mean-test score: ", np.array(scores_test).mean())


def feature_import_analysis(X_train, y_train, model=lin_reg):
    print(X_train.shape)
    X = np.c_[X_trian.values
        ,X_trian[['is_4G_ill','is_arrears','is_m_tour','is_m_WanDa','is_offen_mall','is_m_pay_gym']].values.sum(axis=1).reshape((-1, 1))]
    print(X.shape)
    model.fit(X, y_train)
    data_columns = X_trian.columns.values.tolist()
    data_columns.append("bi_sum")
    # data_columns = np.array(data_columns)
    print(len(model.feature_importances_))
    print(len(data_columns))
    feature_importance = sorted(zip(model.feature_importances_, data_columns), reverse=True)
    for item in feature_importance:
        print(item)


def predict_test(X_train, y_train, X_test, test_id, model=lin_reg):
    model.fit(X_train, y_train)
    print("train score: ", submit_score(y_train, model.predict(X_train)))
    y_pred = model.predict(X_test)
    df_result = pd.DataFrame(np.c_[test_id, y_pred.round()], columns=["id", "score"])
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    file = os.path.join(OUTPUT_PATH, FILE_NAME)
    df_result.to_csv(file, index=False, sep=",", encoding="utf-8")
    print("The predict result was written to ", file)


if __name__ == "__main__":
    userprofile_train = UserprofileData(name="train", is_train=1, file=TRAIN_DATA)
    X_trian, y_train = data_preprocessing(userprofile_train, is_train=1)
    print("X_train shape: ", X_trian.shape)
    print("y_train shape: ", y_train.shape)
    # cross_val(X_trian, y_train, model=gbr_reg)
    feature_import_analysis(X_trian, y_train, model=gbr_reg)

    # userprofile_test = UserprofileData(name="test", is_train=0, file=TEST_DATA)
    # X_test = data_preprocessing(userprofile_test, is_train=0)
    # print("X_test shape: ", X_test.shape)
    # predict_test(X_trian, y_train, X_test, userprofile_test.id, model=gbr_reg)

    # print(X_trian[0])
    # print(X_test[0])

