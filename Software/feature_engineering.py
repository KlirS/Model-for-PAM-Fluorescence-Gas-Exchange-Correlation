#import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler

def create_interaction_features(data, feature_columns):
    """
    Creates interaction features by multiplying pairs of specified features within the given DataFrame.

    Parameters:
    data : DataFrame
        The pandas DataFrame containing the original dataset.
    feature_columns : list of str
        A list of column names for which interaction features will be created.

    Returns:
    DataFrame
        The modified DataFrame with added interaction features for the specified columns.

    This function iterates over all possible pairs of the specified feature columns and
    creates a new feature for each pair by multiplying their values. The name of each
    new feature is constructed by concatenating the names of the multiplied features,
    prefixed with 'Interaction_'.
    """
    fcols = [f for f in feature_columns if f in data.columns] # check which features are in the data columns after cleaning

    for col_pair in itertools.combinations(fcols, 2):
        new_col_name = f'Interaction_{col_pair[0]}_{col_pair[1]}'
        data[new_col_name] = data[col_pair[0]] * data[col_pair[1]]
    return data

def create_new_features(data):

    # combine year and month in an integer
    if "year" in data.columns and "month" in data.columns:
        first_year = data["year"].min()
        data["month_year"] = data[["month", "year"]].apply(lambda col: ((col["year"] - first_year)*12) + col["month"], axis = 1)

    return data

def standardize_features(data, columns_to_scale):
    """
    Standardizes the specified features within the given DataFrame.

    Parameters:
    data : DataFrame
        The pandas DataFrame containing the original dataset.
    columns_to_scale : list of str
        A list of column names whose features will be standardized.

    Returns:
    DataFrame
        The DataFrame with the specified features standardized.
    """
    scaler = StandardScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    return data