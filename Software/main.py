from data_processing import read_data, clean_data, split_data, get_feature_sets
from feature_engineering import standardize_features # create_interaction_features
from model_building import hyperparameter_optimization, plot_feature_importance # perform_cross_validation
from evaluation import evaluate_model, plot_model_performance, eval_model_parameters
import config

def train_and_evaluate_model(X, y, model_type='random_forest', feature_set_name="Feature Set"):
    """
    Trains and evaluates a model based on the given feature set.

    Parameters:
    X : DataFrame
        The feature data.
    y : Series
        The target variable.
    model_type : str, optional
        The type of model to train (default is 'random_forest').
    feature_set_name : str, optional
        The name of the feature set being used (default is "Feature Set").

    Performs hyperparameter optimization, evaluates the model with the test set,
    visualizes model performance, and analyzes feature importance for the minimal feature set.
    """

    # Split the data
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    eval_model_parameters(X_train, y_train, X_test, y_test, feature_set_name, model_type)


    print("Start hyperparameter optimization")

    # Hyperparameter optimization
    best_model, best_mse, best_r2 = hyperparameter_optimization(
        X_train,
        y_train,
        model_type=model_type
    )

    print(f"[Train-Data] Optimized Model ({feature_set_name}): {best_model}, MSE: {best_mse:.4f}, R²: {best_r2:.4f}")

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    test_mse, test_r2 = evaluate_model(
        y_test,
        y_pred,
        f"[Test-Data] Optimized Model ({feature_set_name})"
    )
    print(f"Model train accuracy: {best_model.score(X_train, y_train):.3f}")
    print(f"Model test accuracy: {best_model.score(X_test, y_test):.3f}")

    # Visualize model performance
    plot_model_performance(y_test, y_pred, f"Optimized Model ({feature_set_name})", f"R2={test_r2:.3f}, MSE={test_mse:.3f}")

    # Analyze feature importance, if applicable
    #if feature_set_name == "X_min":
        #feature_names = [
        #    'yii',
        #    'etr',
        #    'par_ges',
        #    'plantage'
        #]
    feature_names = X_train.columns

    plot_feature_importance(
        best_model,
        feature_names,
        feature_set_name=feature_set_name,
        top_n=10 #len(feature_names)
    )

def main():

    # Read and prepare the data
    print("Read data")
    data = read_data(
        config.DATA_FILE_PATH,
        config.SHEET_NAME
    )
    print(f"Prepare data:\n{data.shape}")
    data = clean_data(data)
    data.to_excel("data/12_24 Mais und Basilikum_cleaned.xlsx", sheet_name='Cleaned_Features', header=True, index=False)

    print(f"Data size after cleaning:\n{data.shape}")

    # Define wavelength columns for feature interaction creation
    wavelength_columns = ['405nm', '430nm', '450nm', '465nm', '485nm',
                          '500nm', '527nm', '550nm', '590nm', '630nm',
                          '660nm', '730nm']

    # Obtain various feature sets
    print("Obtain various feature sets")
    X_min, X_max, X_interactions = get_feature_sets(
        data,
        wavelength_columns
    )
    X_interactions.to_excel("data/12_24 Mais und Basilikum_interaction.xlsx", sheet_name='Cleaned_Features', header=True, index=False)


    #data = standardize_features(data, data.columns)
    #X_min = standardize_features(X_min, X_min.columns)
    #X_interactions = standardize_features(X_interactions, X_interactions.columns)

    y = data[config.TARGET_COLUMN]  # Target variable


    # Train and evaluate models for different feature sets
    print("Train and evaluate models for different feature sets")
    print("Train and evaluate: Feature set: X_interactions")
    train_and_evaluate_model(
        X_interactions,
        y,
        model_type=config.USED_MODEL,
        feature_set_name="X_interactions"
    )

    print("Train and evaluate: Feature set: X_min")
    train_and_evaluate_model(
        X_min,
        y,
        model_type=config.USED_MODEL,
        feature_set_name="X_min"
    )

if __name__ == "__main__":
    main()
