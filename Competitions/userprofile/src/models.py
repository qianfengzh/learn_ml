#!/usr/bin/env python
# encoding: utf-8
import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

from estimators import StackingEstimator
from feature_eng import data_preprocessing, UserprofileData
from metrics import submit_score
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.linear_model import LassoLars
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


PROJECT_DIR = '.'
DATA_DIR = r'F:\Git\data\userprofile'
TRAIN_FILE = 'train_dataset.csv'
TEST_FILE = 'test_dataset.csv'
OUTPUT_PATH = os.path.join(DATA_DIR, "result")

TRAIN_DATA = os.path.join(DATA_DIR, TRAIN_FILE)
TEST_DATA = os.path.join(DATA_DIR, TEST_FILE)

FILE_NAME = "lin_reg_base_line-v3-1.csv"


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

gbr_reg = GradientBoostingRegressor(loss='huber', alpha=0.9, learning_rate=0.9, n_estimators=100, max_depth=5
                                        , subsample=0.7, max_features=0.73, random_state=142)

def cross_val(X_train, y_train, model=lin_reg):
    X, y = X_train, y_train
    y_mean = y.mean()
    # print("X shape: ", X.shape)
    # print("y shape: ", y.shape)
    rs = ShuffleSplit(n_splits=5, test_size=.2, random_state=142)
    scores_train = []
    scores_test = []
    for train_idx, test_idx in rs.split(X):
        train_X = X[train_idx]
        # train_y = y[train_idx] - y_mean
        train_y = y[train_idx]
        test_X = X[test_idx]
        test_y = y[test_idx]
        model.fit(train_X, train_y)
        # scores_train.append(submit_score(train_y+y_mean, model.predict(train_X)+y_mean))
        scores_train.append(submit_score(train_y, model.predict(train_X)))
        # scores_test.append(submit_score(test_y, model.predict(test_X)+y_mean))
        scores_test.append(submit_score(test_y, model.predict(test_X)))
        print("=====================")
        print("train score: ", scores_train[-1])
        print("test score: ", scores_test[-1])
    print("mean-train score: ", np.array(scores_train).mean())
    print("mean-test score: ", np.array(scores_test).mean())



def __decomposition(X):
    from sklearn.decomposition import PCA, FastICA, TruncatedSVD
    from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
    n_comp = 5
    # tSVD
    tsvd = TruncatedSVD(n_components=n_comp, random_state=142)
    tsvd_result = tsvd.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_comp, random_state=142)
    pca_result = pca.fit_transform(X)

    # ICA
    ica = FastICA(n_components=n_comp, random_state=142)
    ica_reulst = ica.fit_transform(X)

    # GRP
    grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=142)
    grp_result = grp.fit_transform(X)

    # SRP
    srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=142)
    srp_result = srp.fit_transform(X)

    # Append decomposition components to datasets
    for i in range(1, n_comp + 1):
        # X['pca_' + str(i)] = pca_result[:, i - 1]

        X['ica_' + str(i)] = ica_reulst[:, i - 1]

        # X['grp_' + str(i)] = grp_result[:, i - 1]

        # X['srp_' + str(i)] = srp_result[:, i - 1]

        # X['tsvd_' + str(i)] = tsvd_result[:, i - 1]
    return X


def feature_import_analysis(X_train, y_train, columns=None, model=lin_reg):
    print("origin shape: ",X_train.shape)
    print("add new feature shape: ", X_train.shape)
    model.fit(X_train, y_train)
    data_columns = columns
    # data_columns = X_train.columns.values.tolist()
    # data_columns = X_trian.columns.values.tolist()
    # data_columns.append("bi_sum")
    # for pro_type in ["pca_", "ica_", "grp_", "srp_", "tsvd_"]:
    #     for i in range(1, 13):
    #         data_columns.append(pro_type+str(i))
    print(len(model.feature_importances_))
    print(len(data_columns))
    feature_importance = sorted(zip(model.feature_importances_, data_columns), reverse=True)
    for item in feature_importance:
        print(item)


def add_feature(X):
    print("origin shape: ", X.shape)
    # 新增“bi_sum" feature
    X["bi_sum"] = X[['is_4G_ill', 'is_arrears', 'is_m_tour', 'is_m_WanDa', 'is_offen_mall', 'is_m_pay_gym']].sum(axis=1)
    X["sum_2_to_3"] = X["is_m_tour"] + X["is_m_pay_gym"]
    X["sum_1_to_2"] = X["is_m_movies"] + X["is_blacklist"] + X["is_offen_mall"] + X["is_arrears"]
    X["sum_0_to_1"] = X["is_m_Sam"] + X["is_m_WanDa"] + X["is_real_name"] + X["is_undegraduate"]
    X["sum_m_to_m"] = X["is_m_Sam"] + X["is_m_WanDa"] + X["is_m_tour"] + X["is_m_pay_gym"] + X["is_m_movies"]
    X["sum_p_to_p"] = X["is_blacklist"] + X["is_real_name"] + X["is_undegraduate"] \
                        - X["is_4G_ill"] - X["charge_sensitivity"]
    X["sum_p_certain"] = X["is_blacklist"] + X["is_real_name"] + X["is_undegraduate"] + X["is_arrears"]
    X["sum_customer"] = X["is_m_tour"] + X["is_m_pay_gym"] + X["is_m_movies"] + X["is_offen_mall"] \
                        + X["is_m_Sam"] + X["is_m_WanDa"]

    # X["m_cost_mul_social_persons"] = (X["m_cost"]) * (X["m_social_persons"])
    # X["net_age_per_age"] = (X["net_age_m"] + 1) / (X["age"] * 12 + 50000)
    # X["last_pay_amt_per_m"] = (X["last_pay_amt"] + 1) / (X["last_pay_m"] + 50000)
    # X["m_cost_mul_chargesens"] = (X["m_cost"]) * (X["charge_sensitivity"])
    # X["last_pay_amt_per_6m_consume"] = (X["last_pay_amt"] + 1) / (X["last_6m_avg_consume"] + 50000)


    X["last_pay_m_mul_6m_consume"] = (X["last_pay_m"]) * (X["last_6m_avg_consume"])
    X["m_cost_mul_chargesens"] = (X["m_cost"]) * (X["charge_sensitivity"])
    X["m_balance_mul_chargesens"] = (X["m_balance"]) * (X["charge_sensitivity"])
    X["m_cost_per_social_persons"] = (X["m_cost"]+1) / (X["m_social_persons"]+50000)
    X["net_age_per_age"] = (X["net_age_m"] + 1) / (X["age"] * 12 + 50000)
    X["last_pay_amt_per_m"] = (X["last_pay_amt"]+1) / (X["last_pay_m"] + 50000)
    X["net_age_is_6m"] = X["net_age_m"]>=6
    X["net_age_is_1m"] = X["net_age_m"]>=1
    X["m_cost_much_last6m"] = X["m_cost"] >= X["last_6m_avg_consume"]
    X["m_balance_cover"] = (X["m_balance"]+1) / (X["last_6m_avg_consume"]+ 50000)
    X["movie_video_app"] = X["is_m_movies"] * X["m_video_app_num"]
    X["tour_news_app"] = X["is_m_tour"] * X["m_tour_news_app_num"]
    X["shop_express_num"] = X["m_online_shop_app_num"] + X["m_express_app_num"]
    X["airplane_train_num"] = X["m_airplane_app_num"] + X["m_train_app_num"]

    X = __decomposition(X)
    print("add new feature shape: ", X.shape)
    return X


def grid_search(X, y, model=gbr_reg):
    gbr_param_grid = dict(n_estimators=[300],
                      loss=["huber"],
                      alpha=[0.9],
                      learning_rate=[0.1],
                      subsample=[0.67, 0.7, 0.73],
                      # min_samples_split=[2, 5, 10],
                      # min_samples_leaf=[1, 3, 5],
                      max_depth=[5],
                      max_features=[0.7],
    )
    lasso_param_grid = dict(alpha=[0.1, 0.3, 0.5, 0.9],)
    es_param_grid = dict(alpha=[0.01], l1_ratio=[0.9, 0.95, 0.97] )

    from sklearn.model_selection import GridSearchCV
    score = make_scorer(submit_score, greater_is_better=True)
    # gsearch_gbr = GridSearchCV(estimator=model, scoring=mean_absolute_error, cv=5, param_grid=param_grid)
    gsearch_gbr = GridSearchCV(estimator=model, cv=5, param_grid=es_param_grid, scoring=score)
    gsearch_gbr.fit(X, y)
    return gsearch_gbr.cv_results_ , gsearch_gbr.best_estimator_ , gsearch_gbr.best_score_



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
    y_train = y_train.ravel()
    print("X_train shape: ", X_trian.shape)
    print("y_train shape: ", y_train.shape)
    # X_trian = X_trian.values

    # poly = (PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
    # extra_feature = poly.fit_transform(X_trian[['age', 'net_age_m', 'last_pay_m', 'last_pay_amt', 'last_6m_avg_consume', 'm_cost', 'm_balance'
    #         , 'charge_sensitivity', 'm_social_persons', 'last_3m_mavg_mall', 'm_online_shop_app_num', 'm_express_app_num'
    #         , 'm_finance_app_num', 'm_video_app_num', 'm_airplane_app_num', 'm_train_app_num', 'm_tour_news_app_num']])
    # X_trian = X_trian.values
    # X_trian = np.c_[X_trian, extra_feature]

    X_trian = add_feature(X_trian)
    columns = X_trian.columns.values.tolist()
    # X_trian = X_trian.values
    # 交叉验证
    gbr_reg = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='huber', max_depth=5,
             max_features=0.7, max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=300,
             presort='auto', random_state=142, subsample=0.73, verbose=0,
             warm_start=False)
    rf_reg = RandomForestRegressor(n_estimators=50, max_depth=10)
    xgb_reg = XGBRegressor(booster='gbtree'
                           , n_estimators=200
                           , eta=0.2
                           , max_depth=5
                           , eval_metric='mae', subsample=0.8, colsample_bytree=0.8, seed=142)

    lgb_reg = LGBMRegressor(boosting_type="gbdt"
                            ,objective='regression'
                            ,max_depth=3
                            ,n_estimators=1000
                            ,subsample=0.9
                            ,colsample_bytree=0.9
                            ,num_leaves=50
                            ,random_state=142)

    # X, y = X_trian, y_train
    # rs = ShuffleSplit(n_splits=5, test_size=.2, random_state=142)
    # scores_train = []
    # scores_test = []
    # for train_idx, test_idx in rs.split(X):
    #     train_X = X[train_idx]
    #     train_y = y[train_idx]
    #     test_X = X[test_idx]
    #     test_y = y[test_idx]
    #     gbr_reg.fit(train_X, train_y)
    #     xgb_reg.fit(train_X, train_y)
    #     lgb_reg.fit(train_X, train_y)
    #     train_pred_gbr = gbr_reg.predict(train_X)
    #     train_pred_xgb = xgb_reg.predict(train_X)
    #     train_pred_lgb = lgb_reg.predict(train_X)
    #     train_pred = 0.4*train_pred_gbr + 0.3*train_pred_xgb + 0.3*train_pred_xgb
    #     test_pred_gbr = gbr_reg.predict(test_X)
    #     test_pred_xgb = xgb_reg.predict(test_X)
    #     test_pred_lgb = lgb_reg.predict(test_X)
    #     test_pred = 0.4 * test_pred_gbr + 0.3 * test_pred_xgb + 0.3 * test_pred_lgb
    #     scores_train.append(submit_score(train_y, train_pred))
    #     scores_test.append(submit_score(test_y, test_pred))
    #     print("=====================")
    #     print("train score: ", scores_train[-1])
    #     print("test score: ", scores_test[-1])
    # print("mean-train score: ", np.array(scores_train).mean())
    # print("mean-test score: ", np.array(scores_test).mean())


    ## Stacking model
    # stacked_reg = StackingRegressor(regressors=[lgb_reg, xgb_reg, gbr_reg], meta_regressor=lgb_reg)

    # add PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    # print(X_trian.filter(regex='ica_.*'))
    X_trian = poly.fit_transform(X_trian)
    # X_trian = X_trian.values
    # X_trian = np.c_[X_trian, X_poly]
    print("poly X_train shape: ", X_trian.shape)


    lgb_reg = LGBMRegressor(boosting_type="gbdt"
                            , objective='regression'
                            , max_depth=3
                            , n_estimators=100
                            , subsample=0.9
                            , colsample_bytree=0.9
                            , num_leaves=50
                            , random_state=142)
    cross_val(X_trian, y_train, model=lgb_reg)
    # feature_import_analysis(X_trian, y_train, columns, model=gbr_reg)

    # gsearch 搜索
    # es_reg = ElasticNet(max_iter=1500)
    # cv_results, best_estimator, best_score = grid_search(X_trian, y_train, model = es_reg)
    # cross_val(X_trian, y_train, model=best_estimator)
    # print("best estimator: ", best_estimator)


    # 预测输出
    # gbr_reg = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
    #          learning_rate=0.1, loss='huber', max_depth=5,
    #          max_features=0.7, max_leaf_nodes=None,
    #          min_impurity_decrease=0.0, min_impurity_split=None,
    #          min_samples_leaf=1, min_samples_split=2,
    #          min_weight_fraction_leaf=0.0, n_estimators=300,
    #          presort='auto', random_state=142, subsample=0.73, verbose=0,
    #          warm_start=False)
    # userprofile_test = UserprofileData(name="test", is_train=0, file=TEST_DATA)
    # X_test = data_preprocessing(userprofile_test, is_train=0)
    # X_test = add_feature(X_test).values
    # print("X_test shape: ", X_test.shape)
    # predict_test(X_trian, y_train, X_test, userprofile_test.id, model=gbr_reg)

    # # votting 预测
    # train_X, train_y, test_X = X_trian, y_train, X_test
    # FILE_NAME = "voting_gbr_xgb_lgb_reg_v3-1.csv"
    # gbr_reg.fit(train_X, train_y)
    # xgb_reg.fit(train_X, train_y)
    # lgb_reg.fit(train_X, train_y)
    # train_pred_gbr = gbr_reg.predict(train_X)
    # train_pred_xgb = xgb_reg.predict(train_X)
    # train_pred_lgb = lgb_reg.predict(train_X)
    # train_pred = 0.4 * train_pred_gbr + 0.3 * train_pred_xgb + 0.3 * train_pred_xgb
    # print("train score: ", submit_score(y_train, train_pred))
    # test_pred_gbr = gbr_reg.predict(test_X)
    # test_pred_xgb = xgb_reg.predict(test_X)
    # test_pred_lgb = lgb_reg.predict(test_X)
    # y_pred = 0.4 * test_pred_gbr + 0.3 * test_pred_xgb + 0.3 * test_pred_lgb
    # df_result = pd.DataFrame(np.c_[userprofile_test.id, y_pred.round()], columns=["id", "score"])
    # if not os.path.exists(OUTPUT_PATH):
    #     os.makedirs(OUTPUT_PATH)
    # file = os.path.join(OUTPUT_PATH, FILE_NAME)
    # df_result.to_csv(file, index=False, sep=",", encoding="utf-8")
    # print("The predict result was written to ", file)


