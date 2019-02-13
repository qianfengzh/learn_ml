#!/usr/bin/env python
# encoding: utf-8


from sklearn.metrics import mean_absolute_error


def submit_score(y_true, y_pred):
    return 1/(1+mean_absolute_error(y_true, y_pred))
