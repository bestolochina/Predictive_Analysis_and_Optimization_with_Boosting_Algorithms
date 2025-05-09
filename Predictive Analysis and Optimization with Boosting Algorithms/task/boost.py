import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping

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

if __name__ == '__main__':
    df = download_data().drop_duplicates()

    # Separate the data into the target and features DataFrames;
    features: pd.DataFrame = df.loc[:,['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
    target: pd.Series = df.loc[:,'charges']

    # Perform the absolute z-score calculations and remove the outliers (the threshold is set to 3);
    threshold: int = 3
    z: pd.Series = (target - target.mean()) / target.std()
    mask: pd.Series = z.abs() <= 3

    features = features[mask]
    target = target[mask]

    # Split the data without outliers into training and test sets with train_test_split;
    X_full_train, X_test, y_full_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=True, random_state=10
    )

    # Split the training set into training and validation sets with train_test_split;
    X_train, X_val, y_train, y_val = train_test_split(
        X_full_train, y_full_train, test_size=0.2, shuffle=True, random_state=10
    )

    # Assemble the numerical and categorical transformation steps in a ColumnTransformer.
    # Use StandardScaler for the numerical features and OneHotEncoder for the categorical features;
    numerical_features = ['age', 'bmi', 'children']
    categorical_features = ['sex', 'smoker', 'region']

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
    learning_rate = 1e-1
    max_depth = 10
    n_estimators = 100
    stopping_rounds = 5

    # Evaluation set
    eval_set = [(X_val_transformed, y_val)]
    callbacks = [early_stopping(stopping_rounds=stopping_rounds, verbose=False)]

    # Define models
    models = {
        'xgb_reg': XGBRegressor(
            objective='reg:squarederror',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_jobs=n_jobs,
            verbosity=0
        ),
        'cat_reg': CatBoostRegressor(
            loss_function='RMSE',
            iterations=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            early_stopping_rounds=stopping_rounds,
            silent=True,
            thread_count=n_jobs
        ),
        'lgbm_reg': LGBMRegressor(
            objective='regression',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_jobs=n_jobs,
            verbosity=-1
        )
    }

    # Train and evaluate models
    results = {}

    for name, model in models.items():
        if name == 'lgbm_reg':
            model.fit(
                X_train_transformed, y_train,
                eval_set=eval_set,
                callbacks=callbacks
            )
        else:
            model.fit(
                X_train_transformed, y_train,
                eval_set=eval_set,
                verbose=False
            )

        results[name] = {'model': model}

    # Initialize results dictionary
    mae_results = {
        'xgb_reg': [],
        'cat_reg': [],
        'lgbm_reg': []
    }

    # Evaluate each model on train, val, test sets using MAE
    for name, info in results.items():
        model = info['model']
        for X, y in [(X_train_transformed, y_train), (X_val_transformed, y_val), (X_test_transformed, y_test)]:
            preds = model.predict(X)
            mae = mean_absolute_error(y, preds)
            mae_results[name].append(round(mae, 2))

    # Create DataFrame with the specified index and columns
    mae_df = pd.DataFrame(mae_results, index=['mae_train', 'mae_val', 'mae_test'])

    # Save the DataFrame to CSV
    os.makedirs('../data', exist_ok=True)
    mae_df.to_csv('../data/baseline.csv')

    # Optional: display the DataFrame
    print(mae_df)