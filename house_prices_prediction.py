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

if __name__ == "__main__":
    main()