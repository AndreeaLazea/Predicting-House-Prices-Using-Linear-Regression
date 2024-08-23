"""
house_prices_prediction.py

A simple Python script that predicts house prices based on various features by building a linear regression model.
This script is intended as an introduction to machine learning using Python, focusing on basic concepts such as data preprocessing, model training, and evaluation.

Features:
 - Reads house price data from a CSV file.
 - Handles missing values by removing any incomplete data entries.
 - Normalizes the features (Size, Bedrooms, Age) to standardize the input data.
 - Trains a linear regression model to predict house prices based on the given features.
 - Evaluates the model's performance using Mean Squared Error (MSE).
 - Outputs a comparison of predicted vs. actual house prices to a text file.

Usage:
 - Ensure the dataset is available in a CSV file named 'house_prices.csv' with the following columns:
   * Size: The size of the house in square feet.
   * Bedrooms: The number of bedrooms in the house.
   * Age: The age of the house in years.
   * Price: The selling price of the house (this is the target variable).
 - Run the script in a Python environment:
   `python house_prices_prediction.py`
 - Review the results in the generated 'house_price_predictions.txt' file.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import csv


def handle_missing_values(house_data):
    '''
    Handling missing values, if they exist
    :param house_data: (DataFrame) the house data as pandas DataFrame
    :return: the modified array without the missing data
    '''
    house_data.dropna(inplace=True)


def normalize_features(features):
    """
    Normalizing the features
    :param features: (DateFrame) the DataFrame containing the features to be normalized
    :return: (DataFrame) Normalized features
    """
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    return features

def write_summary_to_file(output_file, target_test, target_pred):
    try:
        with open(output_file, 'w') as f:
            f.write('Predicted vs Actual House Prices\n')
            f.write("================================\n\n")
            for actual, predicted in zip(target_test, target_pred):
                f.write(f"Actual : ${actual:.2f}, Predicted : ${predicted:.2f}\n")
    except Exception as e:
        print("There was an error: ", e)


def main():
    file_path = "house_prices.csv"
    house_data = pd.read_csv(file_path)
    handle_missing_values(house_data)
    print(house_data)
    features = house_data[['Size', 'Bedrooms', 'Age']]
    print(features)
    target = house_data['Price']
    features = normalize_features(features)
    print(features)
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2,
                                                                                random_state=42)

    # training a linear regression model
    model = LinearRegression()
    model.fit(features_train, target_train)

    # predict the house prices on the test set
    target_pred = model.predict(features_test)

    # evaluate the model using Mean Squared Error

    mse = mean_squared_error(target_test, target_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    output_file = "house_price_predictions.txt"
    write_summary_to_file(output_file, target_test, target_pred)



if __name__ == "__main__":
    main()
