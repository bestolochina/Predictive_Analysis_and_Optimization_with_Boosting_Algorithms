import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping
import optuna

def download_data() -> pd.DataFrame:
    data_dir = '../data'
    file_name = 'insurance.csv'
    file_path = os.path.join(data_dir, file_name)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(file_path):
        url = "https://www.dropbox.com/scl/fi/r5033u0e89bpjrk3n9snx/insurance.csv?rlkey=8sv6cnesc6kkqmu6jrizvn9ux&dl=1"
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()  # Raise error if the download failed
        with open(file_path, 'wb') as f:
            f.write(response.content)

    return pd.read_csv(file_path)

def xgb_objective(trial):

    # set the hyperparameters
    xgb_hyperparams = {
  'n_estimators': trial.suggest_int('n_estimators', 25, 1000),
  'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
  'max_depth': trial.suggest_int('max_depth', 2, 20),
  'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0, 1),
  'colsample_bytree': trial.suggest_float('colsample_bytree', 0, 1),
  'objective': trial.suggest_categorical('objective',
                                         ['reg:squarederror', 'reg:gamma', 'reg:absoluteerror', 'reg:tweedie']),
  'alpha': trial.suggest_float('alpha', 0, 5),
  'lambda': trial.suggest_float('lambda', 0, 5),
  'subsample': trial.suggest_float('subsample', 0, 1),
  'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1, 2)
}

    # create an instance of the model with the hyperparameters
    model = XGBRegressor(n_jobs=n_jobs, seed=random_state, verbosity=0,
            early_stopping_rounds=stopping_rounds, **xgb_hyperparams)

    # fit the model on the train set and evaluate on the validation set
    model.fit(X_train_transformed, y_train, eval_set=eval_set, verbose=False)

    # predict with the model
    y_pred = model.predict(X_val_transformed)

    # estimate the evaluation metric
    mae = mean_absolute_error(y_val, y_pred)

    return mae

def cat_objective(trial):

    # set the hyperparameters
    cat_hyperparams = {
  'iterations': trial.suggest_int('iterations', 100, 1000),
  'loss_function': trial.suggest_categorical('loss_function', ['RMSE', 'MAE', 'MAPE', 'Tweedie:variance_power=1.99']),
  'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
  'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0, 1),
  'max_depth': trial.suggest_int('max_depth', 2, 16),
  'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
  'min_child_samples': trial.suggest_int('min_child_samples', 1, 10),
  'subsample': trial.suggest_float('subsample', 0.01, 1),
}

    # create an instance of the model with the hyperparameters
    model = CatBoostRegressor(thread_count=n_jobs, random_seed=random_state, verbose=0,
            early_stopping_rounds=stopping_rounds, **cat_hyperparams)

    # fit the model on the train set and evaluate on the validation set
    model.fit(X_train_transformed, y_train, eval_set=eval_set, verbose=False)

    # predict with the model
    y_pred = model.predict(X_val_transformed)

    # estimate the evaluation metric
    mae = mean_absolute_error(y_val, y_pred)

    return mae

def lgbm_objective(trial):

    # set the hyperparameters
    lgbm_hyperparams = {
  'n_estimators': trial.suggest_int('n_estimators', 25, 1000),
  'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
  'max_depth': trial.suggest_int('max_depth', 2, 20),
  'colsample_bytree': trial.suggest_float('colsample_bytree', 0, 1),
  'objective': trial.suggest_categorical('objective', ['regression', 'gamma']),
  'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
  'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
  'subsample': trial.suggest_float('subsample', 0, 1),
  'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1.0, log=True),
  'min_child_samples': trial.suggest_int('min_child_samples', 2, 30)
}

    # create an instance of the model with the hyperparameters
    model = LGBMRegressor(n_jobs=n_jobs, seed=random_state, verbosity=0,
            **lgbm_hyperparams)

    # fit the model on the train set and evaluate on the validation set
    model.fit(X_train_transformed, y_train, eval_set=eval_set, callbacks=callbacks)

    # predict with the model
    y_pred = model.predict(X_val_transformed)

    # estimate the evaluation metric
    mae = mean_absolute_error(y_val, y_pred)

    return mae

def get_mae(model_) -> list[float]:
    result_ = []
    for X_, y_ in [(X_train_transformed, y_train), (X_val_transformed, y_val), (X_test_transformed, y_test)]:
        prediction = model_.predict(X_)
        mae_ = mean_absolute_error(y_, prediction)
        result_.append(round(mae_, 2))
    return result_

if __name__ == '__main__':
    random_state = 10
    df = download_data().drop_duplicates()

    # Separate the data into the target and features DataFrames;
    features: pd.DataFrame = df.loc[:,['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
    target: pd.Series = df.loc[:,'charges']

    # Perform the absolute z-score calculations and remove the outliers (the threshold is set to 3);
    threshold: int = 3
    z: pd.Series = (target - target.mean()) / target.std()
    mask: pd.Series = z.abs() <= threshold
    features = features[mask]
    target = target[mask]

    # Split the data without outliers into training and test sets with train_test_split;
    X_full_train, X_test, y_full_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=True, random_state=10)

    # Split the training set into training and validation sets with train_test_split;
    X_train, X_val, y_train, y_val = train_test_split(
        X_full_train, y_full_train, test_size=0.2, shuffle=True, random_state=10)

    # Assemble the numerical and categorical transformation steps in a ColumnTransformer.
    # Use StandardScaler for the numerical features and OneHotEncoder for the categorical features;
    numerical_features, categorical_features = ['age', 'bmi', 'children'], ['sex', 'smoker', 'region']

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Fit the ColumnTransformer on the training set;
    preprocessor.fit(X_train)

    # Transform the training, validation, and testing sets with the ColumnTransformer;
    X_train_transformed = preprocessor.transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)

    # Common settings
    n_jobs = -1
    n_trials = 500
    # learning_rate = 1e-1
    # max_depth = 10
    # n_estimators = 100
    stopping_rounds = 5

    # Evaluation set
    eval_set = [(X_val_transformed, y_val)]
    callbacks = [early_stopping(stopping_rounds=stopping_rounds, verbose=False)]

    # Find the optimal hyperparameters for the XGBoost, CatBoost, and LightGBM models;
    xgb_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
    xgb_study.optimize(xgb_objective, n_trials=n_trials, n_jobs=n_jobs)

    cat_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
    cat_study.optimize(cat_objective, n_trials=n_trials, n_jobs=n_jobs)

    lgbm_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
    lgbm_study.optimize(lgbm_objective, n_trials=n_trials, n_jobs=n_jobs)

    # Use the optimized hyperparameters to fit and evaluate XGBoost, CatBoost, and LightGBM models;
    # The optimized hyperparameter is in xgb_study.best_params
    mae_results = {
        'xgb_reg': [],
        'cat_reg': [],
        'lgbm_reg': []
    }
    xgb_model = XGBRegressor(early_stopping_rounds=stopping_rounds, verbosity=0,
                         n_jobs=n_jobs, **xgb_study.best_params)
    xgb_model.fit(X_train_transformed, y_train, eval_set=eval_set, verbose=False)
    mae_results['xgb_reg'] = get_mae(xgb_model)

    cat_model = CatBoostRegressor(early_stopping_rounds=stopping_rounds, verbose=0,
                             thread_count=n_jobs, **cat_study.best_params)
    cat_model.fit(X_train_transformed, y_train, eval_set=eval_set, verbose=False)
    mae_results['cat_reg'] = get_mae(cat_model)

    lgbm_model = LGBMRegressor(verbosity=0, n_jobs=n_jobs, **lgbm_study.best_params)
    lgbm_model.fit(X_train_transformed, y_train, eval_set=eval_set, callbacks=callbacks)
    mae_results['lgbm_reg'] = get_mae(lgbm_model)

    # Create DataFrame with the specified index and columns
    mae_df = pd.DataFrame(mae_results, index=['mae_train', 'mae_val', 'mae_test'])

    # Save the DataFrame to CSV
    os.makedirs('../data', exist_ok=True)
    mae_df.to_csv('../data/optimized.csv')

    # Optional: display the DataFrame
    print(mae_df)

# Plotting
plt.figure(figsize=(10, 6))
mae_df.T.plot(kind='bar', figsize=(10, 6))

plt.title('Mean Absolute Error of Optimized Models')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Model')
plt.xticks(rotation=0)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Dataset', labels=mae_df.index)
plt.tight_layout()
plt.show()
