#!/usr/bin/env python
# encoding: utf-8


from sklearn.model_selection import ShuffleSplit
from feature_eng import data_preprocessing
from metrics import submit_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


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

rf_reg = RandomForestRegressor(n_estimators=10, max_depth=5, criterion="mae", max_features="sqrt")
# featureV1: 常规处理，未增加特征
# n_estimators=10, max_depth=5, criterion="mae", max_features="sqrt"
# train_score: 0.051  test_score: 0.049

def cross_val(model=lin_reg):
    X_prepared, y = data_preprocessing()
    rs = ShuffleSplit(n_splits=5, test_size=.01, random_state=142)
    for train_idx, test_idx in rs.split(X_prepared):
        train_X = X_prepared[train_idx]
        train_y = y[train_idx]
        test_X = X_prepared[test_idx]
        test_y = y[test_idx]

        model.fit(train_X, train_y)
        print("=====================")
        print("train score: ", submit_score(train_y, model.predict(train_X)))
        print("test score: ", submit_score(test_y, model.predict(test_X)))


if __name__ == "__main__":
    cross_val(model=lin_reg)
