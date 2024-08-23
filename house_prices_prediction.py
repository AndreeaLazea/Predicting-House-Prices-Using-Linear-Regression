"""
house_prices_prediction.py


A simple Python script that predicts house prices based on various features by building a simple linear regression model


Features:
 - Reads data from a CSV file.
 -
 -
 -
 -
Usage:
 -
 -
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
