import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    features = {'numerical': numerical_features, 'categorical': categorical_features, 'shape': list(df.shape)}
    print(features)

# Set style
sns.set(style="whitegrid")

# Univariate plots for numerical features
for col in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# Univariate plots for categorical features
for col in categorical_features:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col)
    plt.title(f"Count of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Bivariate plots: Numerical vs Target (assume 'charges' is the target)
target = 'charges'
for col in numerical_features:
    if col != target:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df, x=col, y=target)
        plt.title(f"{col} vs {target}")
        plt.tight_layout()
        plt.show()

# Bivariate plots: Categorical vs Target
for col in categorical_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=col, y=target)
    plt.title(f"{target} by {col}")
    plt.tight_layout()
    plt.show()