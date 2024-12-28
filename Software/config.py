from scipy.stats import randint
import numpy as np


# General configuration for data paths
DATA_FILE_PATH = 'data/12_24 Mais und Basilikum.xlsx'
SHEET_NAME = 'data_all'

# Figure
ENABLE_SAVE_FIGURES = True
ENABLE_SHOW_PLOT = False
FIGURE_FOLDER = 'fig'

# Target variable configuration
TARGET_COLUMN = 'a'

# Use Model:
#USED_MODEL = "random_forest"
USED_MODEL = "xgboost"

# Model parameters for different algorithms
RANDOM_FOREST_PARAMS = {
    'random_state': 42,
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

GAUSSIAN_PROCESS_PARAMS = {
    'random_state': 42,
    'n_restarts_optimizer': 10
}

# Configuration for hyperparameter search ranges
RF_PARAM_DISTRIBUTION = {
    'n_estimators': [100, 200, 300, 400, 500, 800, 1000],
    'max_features': [None, 'sqrt', 'log2'],
    'max_depth': list(range(5, 20)), #[None, 10, 20, 30, 40, 50],
    'min_samples_split': list(np.arange(0.1, 0.3, 0.02)), #list(range(10, 200, 5)), #randint(2, 20), #[2, 5, 10],
    'min_samples_leaf': list(np.arange(0.05, 0.2, 0.02)) #list(range(10, 200, 5)), # randint(1, 20) #[1, 2, 4]
}

XGBOOST_PARAM_DISTRIBUTION = {
    'n_estimators': [100, 200, 350, 500, 600, 700, 1000], # config.RF_PARAM_DISTRIBUTION['n_estimators'],
    'max_depth': [4, 6, 8, 10, 12], # config.RF_PARAM_DISTRIBUTION['max_depth'],
    "learning_rate": [0.01, 0.1],
    "min_child_weight": [ 50, 100, 150, 200, 250, 300, 350], # standardized: 2,5, 10,20,30,
    "gamma": [ 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500], # standardize: 10,20,50,80,  #[ 0.0, 0.1, 0.2], 
    "colsample_bytree":[ 0.8, 0.9, 1.0]
}

# Configuration for splitting the dataset into training and testing sets
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Data cleaning
UNIQUE_FEATURES_IN_COLUMNS = 2
IGNORE_COLUMNS_FOR_DATA_CLEANING = ["year", "m_lightsource"]