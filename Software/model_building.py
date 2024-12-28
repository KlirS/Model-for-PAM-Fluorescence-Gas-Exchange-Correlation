import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
#from scipy.stats import randint
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.model_selection import cross_val_score
#import numpy as np
import matplotlib.pyplot as plt
import config

# def train_test_model(
#        X, 
#        y, 
#        model_type='random_forest', 
#        test_size=0.3, 
#        random_state=42
#    ):
    # """
    # Trains and tests a model based on the specified model type and data. 

    # Parameters:
    # X : DataFrame
    #     My feature data for training and testing.
    # y : Series
    #     The target variable for training and testing.
    # model_type : str, optional
    #     The type of model to train ('random_forest' or 'gaussian_process').
    #     Default is 'random_forest'.
    # test_size : float, optional
    #     The proportion of the dataset to include in the test split. Default is 0.3.
    # random_state : int, optional
    #     Controls the shuffling applied to the data before applying the split.
    #     Default is 42.

    # Returns:
    # tuple
    #     A tuple containing the trained model, mean squared error (MSE), and R-squared (R²) score.
    # """

    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, 
    #     y, 
    #     test_size=test_size, 
    #     random_state=random_state
    # )
    
    # # Initialize and train the specified model
    # if model_type == 'random_forest':
    #     model = RandomForestRegressor(random_state=random_state)
    # elif model_type == 'gaussian_process':
    #     kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1)
    #     model = GaussianProcessRegressor(
    #         kernel=kernel, 
    #         n_restarts_optimizer=10, 
    #         random_state=random_state
    #     )
    # else:
    #     raise ValueError("Unsupported model type")

    # model.fit(X_train, y_train)
    
    # # Make predictions and evaluate the model
    # y_pred = model.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    # return model, mse, r2

def hyperparameter_optimization(
        X, 
        y, 
        model_type='random_forest', 
        n_iter=100, 
        cv=5, 
        random_state=config.RANDOM_STATE,
        test_size=config.TEST_SIZE
    ):
    """
    Performs hyperparameter optimization for the specified model type using Randomized Search CV.

    Parameters:
    X : DataFrame
        The feature data for training.
    y : Series
        The target variable for training.
    model_type : str, optional
        The type of model to perform optimization on ('random_forest' is currently supported).
        Default is 'random_forest'.
    n_iter : int, optional
        Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
        Default is 100.
    cv : int, optional
        Number of folds in cross-validator.
        Default is 5.
    random_state : int, optional
        Controls the random seed given to the method chosen.
        Default is 42.

    Returns:
    tuple
        A tuple containing the optimized model, mean squared error (MSE), and R-squared (R²) score.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Define parameter distribution and initialize RandomizedSearchCV based on the model type
    if model_type == 'random_forest':
        param_dist = {
            'n_estimators': config.RF_PARAM_DISTRIBUTION['n_estimators'],
            'max_features': config.RF_PARAM_DISTRIBUTION['max_features'],
            'max_depth': config.RF_PARAM_DISTRIBUTION['max_depth'],
            'min_samples_split': config.RF_PARAM_DISTRIBUTION['min_samples_split'],
            'min_samples_leaf': config.RF_PARAM_DISTRIBUTION['min_samples_leaf']
        }
        random_search = RandomizedSearchCV(
            RandomForestRegressor(random_state=random_state),
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1,
            random_state=random_state
        )
    elif model_type == 'xgboost':
        import xgboost
        param_dist = {
            'n_estimators': config.XGBOOST_PARAM_DISTRIBUTION['n_estimators'],
            'max_depth': config.XGBOOST_PARAM_DISTRIBUTION['max_depth'],
            "learning_rate": config.XGBOOST_PARAM_DISTRIBUTION['learning_rate'],
            "min_child_weight": config.XGBOOST_PARAM_DISTRIBUTION['min_child_weight'],
            "gamma": config.XGBOOST_PARAM_DISTRIBUTION['gamma'],
            "colsample_bytree": config.XGBOOST_PARAM_DISTRIBUTION['colsample_bytree'],
        }
        random_search = RandomizedSearchCV(
            xgboost.XGBRegressor(random_state=random_state, eval_metric='rmsle'),
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            verbose=True
        )
    else:
        raise ValueError("Unsupported model type or optimization method")

    # Perform the random search and fit the model
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    #print(best_model.estimators_)
    
    # Evaluate the optimized model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return best_model, mse, r2

def perform_cross_validation(model, X, y, cv=5, scoring='r2'):
    """
    Performs cross-validation on the given model and dataset.

    Parameters:
    model : estimator object implementing 'fit'
        The object to use to fit the data. This is the model that will be evaluated.
    X : array-like, shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.
    y : array-like, shape (n_samples,) or (n_samples, n_outputs)
        The target variable to try to predict in the case of supervised learning.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy. Default is 5.
    scoring : str, callable, list/tuple, dict or None, optional
        A single string or a callable to evaluate the predictions on the test set.
        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.
        Default is 'r2' for the R^2 score.

    Returns:
    dict
        A dictionary with the mean and standard deviation of the scores across all cross-validation folds.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {
        'mean_score': scores.mean(),
        'std_score': scores.std()
    }

def plot_feature_importance(model, feature_names, feature_set_name="default", top_n=10):
    """
    Plots the top N feature importances for the given model.

    Parameters:
    model : model object
        The trained model object that must have a `feature_importances_` attribute.
    feature_names : list of str
        A list containing the names of the features corresponding to the model's feature importances.
    top_n : int, optional
        The number of top features to plot. Default is 10.

    If the model does not have a `feature_importances_` attribute, the function prints a message and exits.
    """

    if not hasattr(model, 'feature_importances_'):
        print("The chosen model does not provide feature importance information.")
        return
    
    #importances = model.feature_importances_
    feat_importances = pd.Series(model.feature_importances_, index=feature_names)

    #indices = np.argsort(importances)[-top_n:]
    feat_importances = feat_importances.nlargest(top_n)
    feat_importances = feat_importances.sort_values(ascending=True)


    if config.USED_MODEL == "xgboost":
        col1 = "cornflowerblue"
        col2 = "navy"
    else:
        col1 = "lightseagreen"
        col2 = "darkgreen"

    # Plot the top N feature importances
    plt.figure(figsize=(10, 6))
    plt.title(f"Top-{top_n} Feature Importances: Dataset {feature_set_name}")
    # Daten sortieren, sodass der höchste Wert links steht
    feat_importances = feat_importances.rename(index={"plantage": "plant_age"})
    feat_importances = feat_importances.rename(index={"par_ges": "par_total"})
    feat_importances = feat_importances.rename(index={"plantid": "plant_id"})
    feat_importances_sorted = feat_importances.sort_values(ascending=False)

    feat_importances_sorted.plot(kind='bar', color=col2, align='center')
    #plt.barh(
    #    range(top_n), 
    #    importances[indices], 
    #    color='blue', 
    #    align='center'
    #)
    
    #plt.yticks(
    #    range(top_n), 
    #    [feature_names[i] for i in indices]
    #)

    # Design-Anpassungen
    plt.gca().spines['top'].set_visible(False)   # Obere Linie entfernen
    plt.gca().spines['right'].set_visible(False) # Rechte Linie entfernen
    plt.gca().spines['left'].set_linewidth(0.8)  # Linke Achse schmaler
    plt.gca().spines['bottom'].set_linewidth(0.8) # Untere Achse schmaler
    plt.gca().tick_params(axis='both', direction='out', length=6, width=0.8, labelsize=15) # Achsen-Ticks anpassen


    plt.ylabel("Relative Feature Importance", fontsize=15, labelpad=10)
    plt.xlabel("Feature", fontsize=15, labelpad=10)
    if config.ENABLE_SAVE_FIGURES: plt.savefig(f'{config.FIGURE_FOLDER}/{config.USED_MODEL}_{feature_set_name}_feature_importance.png', bbox_inches='tight')
    if config.ENABLE_SHOW_PLOT: plt.show()
