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

        if 'optimized.csv' not in os.listdir('../data'):
            return CheckResult.wrong("There is no file called optimized.csv in the data directory")

        return CheckResult.correct()

    @dynamic_test()
    def test2(self):

        df = pd.read_csv('../data/optimized.csv', index_col=0)

        if df.shape != (3, 3):
            return CheckResult.wrong("The optimized.csv file should have 3 rows and 3 columns")

        cols = sorted([i.lower() for i in list(df.columns)])
        indices = sorted([i.lower() for i in list(df.index)])

        if cols != ['cat_reg', 'lgbm_reg', 'xgb_reg']:
            return CheckResult.wrong("Check the titles of your columns in optimized.csv")

        if indices != ['mae_test', 'mae_train', 'mae_val']:
            return CheckResult.wrong("The titles of your indices should be `mae_train`, `mae_val`, 'mae_test`")

        df.columns = [i.lower() for i in list(df.columns)]
        df.index = [i.lower() for i in list(df.index)]

        xgb_train_mae = df.loc['mae_train', 'xgb_reg']
        xgb_valid_mae = df.loc['mae_val', 'xgb_reg']
        xgb_test_mae = df.loc['mae_test', 'xgb_reg']

        if abs(xgb_valid_mae - xgb_train_mae) > 700:
            return CheckResult.wrong("Large difference between XGB mae_train and mae_val")

        if abs(xgb_test_mae - xgb_train_mae) > 700:
            return CheckResult.wrong("Large difference between XGB mae_train and mae_test")

        cat_train_mae = df.loc['mae_train', 'cat_reg']
        cat_valid_mae = df.loc['mae_val', 'cat_reg']
        cat_test_mae = df.loc['mae_test', 'cat_reg']

        if abs(cat_valid_mae - cat_train_mae) > 700:
            return CheckResult.wrong("Large difference between CatBoost mae_train and mae_val")

        if abs(cat_test_mae - cat_train_mae) > 400:
            return CheckResult.wrong("Large difference between CatBoost mae_train and mae_test")

        lgbm_train_mae = df.loc['mae_train', 'lgbm_reg']
        lgbm_valid_mae = df.loc['mae_val', 'lgbm_reg']
        lgbm_test_mae = df.loc['mae_test', 'lgbm_reg']

        if abs(lgbm_valid_mae - lgbm_train_mae) > 700:
            return CheckResult.wrong("Large difference between LGBM mae_train and mae_val")

        if abs(lgbm_test_mae - lgbm_train_mae) > 700:
            return CheckResult.wrong("Large difference between LGBM mae_train and mae_test")

        df.T.plot(
            kind='bar', figsize=(10, 4)
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('../data/stage5-fig.png')

        return CheckResult.correct()
