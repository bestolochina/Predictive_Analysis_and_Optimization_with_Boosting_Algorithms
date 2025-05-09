import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

    # Print the dictionary containing the shapes of the transformed training, validation, and testing sets.
    print(
        {
            'train': list(X_train_transformed.shape),
            'validation': list(X_val_transformed.shape),
            'test': list(X_test_transformed.shape)
        }
    )
