import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import config

def eval_model_parameter(x_train, y_train, x_test, y_test, feature_set_name, param_dict, parameter_name, model, valuerange=(0,1), dtype=float):
    import numpy as np
    import matplotlib.patches as mpatches

    if config.USED_MODEL == "xgboost":
        col1 = "cornflowerblue"
        col2 = "navy"
    else:
        col1 = "lightseagreen"
        col2 = "darkgreen"

    plt.figure()
    #plt.scatter(x_test, y_test, s=20, edgecolor="black", c="darkorange", label="data")

    parameters = np.linspace(valuerange[0], valuerange[1], valuerange[2], endpoint=True, dtype=dtype)
    for i, parameter in enumerate(parameters):
        m = model( **{parameter_name:parameter}, **param_dict, n_jobs=-1)
        m.fit(x_train, y_train)   
        plt.plot(parameter, m.score(x_test, y_test), color=col2, marker='o', label="Test")
        plt.plot(parameter, m.score(x_train, y_train), color=col1, marker='o', label="Training")

    # Design-Anpassungen
    plt.gca().spines['top'].set_visible(False)   # Obere Linie entfernen
    plt.gca().spines['right'].set_visible(False) # Rechte Linie entfernen
    plt.gca().tick_params(axis='both', labelsize=15)  # Tick-Schriftgröße erhöhen

    

    plt.xlabel(f"Parameter: {parameter_name}", fontsize=15)
    plt.ylabel(r"$R^2$", fontsize=15)
    plt.title(f"Feature Influence for Feature: {feature_set_name}\n {param_dict}\n")
    # Create a legend with a color box
    test_patch = mpatches.Patch(color=col2, label='Test')
    train_patch = mpatches.Patch(color=col1, label='Train')
    plt.legend(handles=[test_patch, train_patch], loc='upper right', framealpha=0.5, frameon=True, fontsize=15)
    if config.ENABLE_SAVE_FIGURES: plt.savefig(f'{config.FIGURE_FOLDER}/{config.USED_MODEL}_{feature_set_name}_{parameter_name}.png', bbox_inches='tight')
    if config.ENABLE_SHOW_PLOT: plt.show()

def eval_model_parameters(X_train, y_train, X_test, y_test, feature_set_name, model_type):
    from evaluation import eval_model_parameter
    print(f"Model selected: {model_type}")
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor

        print("Evaluate: max_depth")
        param_dict = {"n_estimators":200, "min_samples_split":0.23, "min_samples_leaf":0.05}
        parameter_name = "max_depth"
        valuerange = (4,10, 4)
        eval_model_parameter(X_train, y_train, X_test, y_test, feature_set_name, param_dict, parameter_name, model, valuerange, int)

        print("Evaluate: n_estimators")
        param_dict = {"max_depth":15, "min_samples_split":0.23, "min_samples_leaf":0.05}
        parameter_name = "n_estimators"
        valuerange = (100, 1000, 10)
        eval_model_parameter(X_train, y_train, X_test, y_test, feature_set_name, param_dict, parameter_name, model, valuerange, int)
       
        print("Evaluate: min_samples_split")
        param_dict = {"max_depth":15, "n_estimators":200, "min_samples_leaf":0.05}
        parameter_name = "min_samples_split"
        valuerange = (0.01, 0.4, 20)
        eval_model_parameter(X_train, y_train, X_test, y_test, feature_set_name, param_dict, parameter_name, model, valuerange, float)
        
        print("Evaluate: min_samples_leaf")
        param_dict = {"max_depth":15, "n_estimators":200, "min_samples_split":0.23}
        parameter_name = "min_samples_leaf"
        valuerange = (0.01, 0.4, 20)
        eval_model_parameter(X_train, y_train, X_test, y_test, feature_set_name, param_dict, parameter_name, model, valuerange, float)

    elif model_type == 'xgboost':
        import xgboost
        model = xgboost.XGBRegressor
        # https://xgboost.readthedocs.io/en/stable/parameter.html

        print("Evaluate: max_depth")
        param_dict = {"n_estimators":800, "learning_rate":0.1, "min_child_weight":50, "gamma":100, "colsample_bytree":1.0}
        parameter_name = "max_depth"
        valuerange = (4,12, 4)
        eval_model_parameter(X_train, y_train, X_test, y_test, feature_set_name, param_dict, parameter_name, model, valuerange, int)

        print("Evaluate: n_estimators")
        param_dict = {"max_depth":4, "learning_rate":0.1, "min_child_weight":50, "gamma":100, "colsample_bytree":1.0}
        parameter_name = "n_estimators"
        valuerange = (200,1000, 20)
        eval_model_parameter(X_train, y_train, X_test, y_test, feature_set_name, param_dict, parameter_name, model, valuerange, int)

        print("Evaluate: gamma")
        param_dict = {"n_estimators": 800, "max_depth":4, "learning_rate":0.1, "min_child_weight":50, "colsample_bytree":1.0}
        parameter_name = "gamma"
        valuerange = (0,5000, 30)
        eval_model_parameter(X_train, y_train, X_test, y_test, feature_set_name, param_dict, parameter_name, model, valuerange, int)

        print("Evaluate: min_child_weight")
        param_dict = {"n_estimators": 800, "max_depth":4, "learning_rate":0.1, "gamma":100, "colsample_bytree":1.0}
        parameter_name = "min_child_weight"
        valuerange = (1,1000, 20)
        eval_model_parameter(X_train, y_train, X_test, y_test, feature_set_name, param_dict, parameter_name, model, valuerange, int)

        print("Evaluate: colsample_bytree")
        param_dict = {"n_estimators": 800, "max_depth":4, "learning_rate":0.1, "gamma":100, "min_child_weight":50}
        parameter_name = "colsample_bytree"
        valuerange = (0.7,1.0, 4)
        eval_model_parameter(X_train, y_train, X_test, y_test, feature_set_name, param_dict, parameter_name, model, valuerange, float)

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluates the performance of a model by calculating the Mean Squared Error (MSE) and R-squared (R²) score.

    Parameters:
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values by the model.
    model_name : str, optional
        The name of the model being evaluated (default is "Model").

    Returns:
    tuple
        A tuple containing the MSE and R² score of the model.
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.2f}, R²: {r2:.2f}")
    return mse, r2

def plot_model_performance(y_true, y_pred, model_name="Model", add_titel=""):
    """
    Plots the performance of a model by comparing the true target values against the predicted values.

    Parameters:
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values by the model.
    model_name : str, optional
        The name of the model being evaluated (default is "Model").

    This function creates a scatter plot of the true vs predicted values and a line representing
    the perfect predictions for visual comparison.
    """

    if config.USED_MODEL == "xgboost":
        col1 = "cornflowerblue"
        col2 = "navy"
    else:
        col1 = "lightseagreen"
        col2 = "darkgreen"
    

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color=col2)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], linestyle='--', color=col1, lw=2)
    plt.gca().spines['top'].set_visible(False)   # Obere Linie entfernen
    plt.gca().spines['right'].set_visible(False) # Rechte Linie entfernen
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predicted Values', fontsize=15)
    plt.title(f'{model_name} Performance {add_titel}')
    if config.ENABLE_SAVE_FIGURES: plt.savefig(f'{config.FIGURE_FOLDER}/{model_name}_Performance.png', bbox_inches='tight')
    if config.ENABLE_SHOW_PLOT: plt.show()

    errors = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color=col1, edgecolor=col2, alpha=0.7)
    plt.axvline(0, color='slategray', linestyle='--', lw=2)  # Vertikale Linie bei 0
    plt.gca().spines['top'].set_visible(False)   # Obere Linie entfernen
    plt.gca().spines['right'].set_visible(False) # Rechte Linie entfernen
    plt.xlabel('Error (True - Predicted)', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.title(f'Error Frequency Distribution ({model_name})')
    if config.ENABLE_SAVE_FIGURES: plt.savefig(f'{config.FIGURE_FOLDER}/{model_name}_error_histogram.png', bbox_inches='tight')
    if config.ENABLE_SHOW_PLOT: plt.show()