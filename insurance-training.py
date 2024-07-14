import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def main(args):
    # function that reads the data
    df = get_data(args.input_data)

    # Clean the data for duplicates and outliers
    new_data = clean_data(df)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = split_data(new_data)

    # Input data into model
    data_model(X_train, X_test, y_train, y_test, args)


# Function that reads the data
def get_data(path):
    print("Reading data ...")
    df = pd.read_csv(path)
    return df


def clean_data(data):
    df_copy = data.copy()

    # Check for duplicates
    duplicates = df_copy.duplicated()
    print(f'Number of duplicate rows: {duplicates.sum()}')

    # Remove duplicate rows
    df_copy = df_copy.drop_duplicates()

    # Retrieve index location for prices greater than 50k and drop
    high_price = df_copy[df_copy['charges'] >= 50000].index
    df_copy.drop(high_price, axis=0, inplace=True)

    # Label encoding the categorical columns
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()

    # Fit and transform the 'sex' and 'smoker' columns
    df_copy['sex'] = le_sex.fit_transform(df_copy['sex'])
    df_copy['smoker'] = le_smoker.fit_transform(df_copy['smoker'])

    # Apply pd.get_dummies to encode the 'region' column
    df_encoded = pd.get_dummies(df_copy, columns=['region'], dtype=float)

    return df_encoded


def split_data(cleaned_data):
    y = cleaned_data['charges'].values
    cleaned_data.drop('charges', axis=1, inplace=True)
    X = cleaned_data.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def data_model(X_train, X_test, y_train, y_test, args):
    # Start a new MLflow run
    with mlflow.start_run():
        # Model parameters
        n_estimators = args.n_estimators
        learning_rate = args.learning_rate
        max_depth = args.max_depth
        random_state = args.random_state

        # Initialize and train the GradientBoostingRegressor
        gbr = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
        model = gbr.fit(X_train, y_train)

        # Save the model
        # Log the model with MLflow
        mlflow.sklearn.log_model(model, "model")

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        mean_abs_error = mean_absolute_error(y_test, y_pred)
        mean_sq_error = mean_squared_error(y_test, y_pred)
        root_mean_sq_error = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Plot predicted vs actual values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7, s=100)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        plt.savefig('Regression-line.png')

        # Log parameters and metrics
        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "random_state": random_state
        }
        mlflow.log_params(params)

        metrics = {
            "mean_absolute_error": mean_abs_error,
            "mean_squared_error": mean_sq_error,
            "root_mean_squared_error": root_mean_sq_error,
            "R2": r2
        }
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("Regression-line.png")


def parse_args():
    # Setup arg parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--input_data", dest='input_data', type=str, required=True)
    parser.add_argument("--n_estimators", dest='n_estimators', type=int, default=100)
    parser.add_argument("--learning_rate", dest='learning_rate', type=float, default=0.1)
    parser.add_argument("--max_depth", dest='max_depth', type=int, default=3)
    parser.add_argument("--random_state", dest='random_state', type=int, default=123)

    # Parse args
    args = parser.parse_args()

    # Return args
    return args


# Run script
if __name__ == "__main__":
    # Add space in logs
    print("\n\n")
    print("*" * 60)

    # Parse args
    args = parse_args()

    # Run main function
    main(args)

    # Add space in logs
    print("*" * 60)
    print("\n\n")

