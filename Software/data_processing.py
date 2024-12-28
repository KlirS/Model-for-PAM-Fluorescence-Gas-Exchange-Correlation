import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from feature_engineering import create_interaction_features, create_new_features
import config

def read_data(
        file_path, 
        sheet_name='data_all'
    ):
    """
    Reads data from my Excel file into a pandas DataFrame.

    Parameters:
    file_path : str
        The path to my Excel file.
    sheet_name : str, optional
        The name of the sheet to read (default is 'data_all').

    Returns:
    DataFrame
        The data read from the specified Excel sheet.
    """
    return pd.read_excel(
        file_path, 
        sheet_name=sheet_name
    )


def clean_data(data):
    """
    Cleans the provided DataFrame by removing rows with missing values.

    Parameters:
    data : DataFrame
        The pandas DataFrame to be cleaned.

    Returns:
    DataFrame
        The cleaned DataFrame with no rows containing missing values.
    """
    if "cond" in data.columns:
        data = data.drop("cond", axis=1)
    column_list = list(data.columns)
    #column_list.remove("cond")
    data = data.dropna(subset=column_list)

    data = data.drop_duplicates()

    ignore_columns = config.IGNORE_COLUMNS_FOR_DATA_CLEANING
    col_lst = [f for f in data.columns if f not in ignore_columns]
    data_cleaned = data[col_lst].loc[:,data[col_lst].apply(pd.Series.nunique) > config.UNIQUE_FEATURES_IN_COLUMNS] # drop all columns with less than 3 unique values
    dropped_cols = [f for f in data.columns if f not in data_cleaned.columns]
    for f in ignore_columns:
        data_cleaned[f] = data[f]

    print(f"Dropped columns: {dropped_cols}")


    # steady plant id
    pid = data_cleaned["plantid"].unique()
    data_cleaned["plantid"] = data_cleaned["plantid"].replace(pid,range(1,len(pid)+1))

    return data_cleaned

def split_data(
        X, 
        y, 
        test_size=0.3, 
        random_state=42
    ):
    """
    Splits my dataset into training and testing data sets.

    Parameters:
    X : DataFrame or ndarray
        The feature dataset.
    y : Series or ndarray
        The target variable.
    test_size : float, optional
        The proportion of the dataset to include in the test split (default is 0.3).
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split (default is 42).

    Returns:
    X_train, X_test, y_train, y_test : tuple of DataFrames or ndarrays
        The split datasets for training and testing.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def get_feature_sets(data, wavelength_columns):
    """
    Generates different feature sets from the provided data, including a minimal set,
    a maximal set, and a set with interaction features based on specified wavelength columns.

    Parameters:
    data : DataFrame
        The pandas DataFrame containing my original dataset.
    wavelength_columns : list of str
        A list of column names representing wavelength features to be used for creating interaction features.

    Returns:
    tuple of DataFrame
        A tuple containing three DataFrames: X_min (minimal feature set), X_max (maximal feature set),
        and X_interactions (feature set with interaction features).
    """
    
    # Minimal feature set
    X_min = data[
        [
            'yii', 
            'etr', 
            'par_ges', 
            'plantage'
        ]
    ]
    
    # Maximal feature set 
    X_max = data.drop(
        [config.TARGET_COLUMN], 
        axis=1
    )
    
    # Feature set with interaction features
    # First, create a copy of X_max
    X_interactions = X_max.copy()

    # Then, add the interaction features
    X_interactions = create_interaction_features(
        X_interactions, 
        wavelength_columns
    )
    X_interactions = create_new_features(X_interactions)
    
    return X_min, X_max, X_interactions
