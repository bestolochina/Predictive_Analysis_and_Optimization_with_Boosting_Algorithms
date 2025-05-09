from hstest import StageTest, CheckResult, dynamic_test
from hstest.stage_test import List
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BoostingTest(StageTest):

    @dynamic_test()
    def test1(self):
        if not os.path.exists('../data'):
            return CheckResult.wrong("There is no directory called data")

        if 'insurance.csv' not in os.listdir('../data'):
            return CheckResult.wrong("There is no file called insurance.csv")

        if 'baseline.csv' not in os.listdir('../data'):
            return CheckResult.wrong("There is no file called baseline.csv in the data directory")

        return CheckResult.correct()

    @dynamic_test()
    def test2(self):

        df = pd.read_csv('../data/baseline.csv', index_col=0)

        if df.shape != (3, 3):
            return CheckResult.wrong("The baseline.csv file should have 3 rows and 3 columns")

        cols = sorted([i.lower() for i in list(df.columns)])
        indices = sorted([i.lower() for i in list(df.index)])

        if cols != ['cat_reg', 'lgbm_reg', 'xgb_reg']:
            return CheckResult.wrong("Check the titles of your columns in baseline.csv")

        if indices != ['mae_test', 'mae_train', 'mae_val']:
            return CheckResult.wrong("The titles of your indices should be `mae_train`, `mae_val`, 'mae_test`")

        df.columns = [i.lower() for i in list(df.columns)]
        df.index = [i.lower() for i in list(df.index)]

        xgb_train_mae = df.loc['mae_train', 'xgb_reg']
        xgb_valid_mae = df.loc['mae_val', 'xgb_reg']
        xgb_test_mae = df.loc['mae_test', 'xgb_reg']

        if xgb_train_mae < 129 or xgb_train_mae > 158:
            return CheckResult.wrong("XGB mae_train result is incorrect")

        if xgb_valid_mae < 2648 or xgb_valid_mae > 3248:
            return CheckResult.wrong("XGB mae_val result is incorrect")

        if xgb_test_mae < 2598 or xgb_test_mae > 3174:
            return CheckResult.wrong("XGB test_mae result is incorrect")

        cat_train_mae = df.loc['mae_train', 'cat_reg']
        cat_valid_mae = df.loc['mae_val', 'cat_reg']
        cat_test_mae = df.loc['mae_test', 'cat_reg']

        if cat_train_mae < 1818 or cat_train_mae > 2220:
            return CheckResult.wrong("CatBoost mae_train result is incorrect")

        if cat_valid_mae < 2383 or cat_valid_mae > 2911:
            return CheckResult.wrong("CatBoost mae_val result is incorrect")

        if cat_test_mae < 2500 or cat_test_mae > 3054:
            return CheckResult.wrong("CatBoost test_mae result is incorrect")

        lgbm_train_mae = df.loc['mae_train', 'lgbm_reg']
        lgbm_valid_mae = df.loc['mae_val', 'lgbm_reg']
        lgbm_test_mae = df.loc['mae_test', 'lgbm_reg']

        if lgbm_train_mae < 1740 or lgbm_train_mae > 2126:
            return CheckResult.wrong("LGBM mae_train result is incorrect")

        if lgbm_valid_mae < 2223 or lgbm_valid_mae > 2717:
            return CheckResult.wrong("LGBM mae_val result is incorrect")

        if lgbm_test_mae < 2466 or lgbm_test_mae > 3012:
            return CheckResult.wrong("LGBM test_mae result is incorrect")

        return CheckResult.correct()
