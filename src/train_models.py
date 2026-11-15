import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from src.data_preprocessing import prepare_data, preprocessor
def prepare_simple_data(file_path):
    df = pd.read_csv(file_path)

    X = df[["median_income"]]
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test
def prepare_polynomial_data(file_path):
    df = pd.read_csv(file_path)

    X = df[["median_income"]]
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test
def prepare_svr_data(file_path):
    df = pd.read_csv(file_path)

    X = df[["median_income"]]
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test
def prepare_decision_tree_data(file_path):
    X_train, X_test, y_train, y_test = prepare_data(file_path)

    return X_train, X_test, y_train, y_test
def prepare_random_forest_data(file_path):
    X_train, X_test, y_train, y_test = prepare_data(file_path)

    return X_train, X_test, y_train, y_test
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model
def tune_random_forest(X_train, y_train):

    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, 20],
        "min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    return grid_search
def run_random_forest_tuning():

    X_train, X_test, y_train, y_test = prepare_random_forest_data(
        "data/raw/housing.csv"
    )

    grid_search = tune_random_forest(X_train, y_train)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross Validation R2 Score:")
    print(grid_search.best_score_)

    best_model = grid_search.best_estimator_
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    print("Preprocessor saved successfully!")

    predictions = best_model.predict(X_test)

    results = evaluate_model(y_test, predictions)

    print("\nTuned Random Forest Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
        best_model = grid_search.best_estimator_
        joblib.dump(best_model, "models/random_forest_model.pkl")

print("\nBest model saved successfully!")
def predict_random_forest(model, X_test):
    predictions = model.predict(X_test)
    return predictions
def run_random_forest():
    X_train, X_test, y_train, y_test = prepare_random_forest_data(
        "data/raw/housing.csv"
    )

    model = train_random_forest(X_train, y_train)

    predictions = predict_random_forest(model, X_test)

    results = evaluate_model(y_test, predictions)

    print("\nRandom Forest Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model
def predict_decision_tree(model, X_test):
    predictions = model.predict(X_test)
    return predictions
def run_decision_tree():
    X_train, X_test, y_train, y_test = prepare_decision_tree_data(
        "data/raw/housing.csv"
    )

    model = train_decision_tree(X_train, y_train)

    predictions = predict_decision_tree(model, X_test)

    results = evaluate_model(y_test, predictions)

    print("\nDecision Tree Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
def train_svr(X_train, y_train):
    model = SVR(
        kernel="rbf",
        C=100000,
        epsilon=10000
    )

    model.fit(X_train, y_train)

    return model
def scale_svr_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
def predict_svr(model, X_test_scaled):
    predictions = model.predict(X_test_scaled)

    return predictions
def run_svr():
    X_train, X_test, y_train, y_test = prepare_svr_data(
        "data/raw/housing.csv"
    )

    X_train_scaled, X_test_scaled, scaler = scale_svr_data(
        X_train, X_test
    )

    model = train_svr(X_train_scaled, y_train)

    predictions = predict_svr(model, X_test_scaled)

    results = evaluate_model(y_test, predictions)

    print("\nSVR Results:")

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
def create_polynomial_features(X_train, X_test):
    poly = PolynomialFeatures(degree=2)

    X_train_poly = poly.fit_transform(X_train)

    X_test_poly = poly.transform(X_test)

    return X_train_poly, X_test_poly, poly
def train_polynomial_regression(X_train_poly, y_train):
    model = LinearRegression()

    model.fit(X_train_poly, y_train)

    return model
def predict_polynomial(model, X_test_poly):
    predictions = model.predict(X_test_poly)

    return predictions
def run_polynomial_regression():
    X_train, X_test, y_train, y_test = prepare_polynomial_data(
        "data/raw/housing.csv"
    )

    X_train_poly, X_test_poly, poly = create_polynomial_features(
        X_train, X_test
    )

    model = train_polynomial_regression(
        X_train_poly, y_train
    )

    predictions = predict_polynomial(model, X_test_poly)

    results = evaluate_model(y_test, predictions)

    print("\nPolynomial Regression Results:")

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
def train_simple_linear_regression(X_train, y_train):
    model = LinearRegression()

    model.fit(X_train, y_train)

    return model
def predict_simple(model, X_test):
    predictions = model.predict(X_test)

    return predictions
from src.data_preprocessing import prepare_data
def train_linear_regression(X_train, y_train):
    model = LinearRegression()

    model.fit(X_train, y_train)

    return model
def predict(model, X_test):
    predictions = model.predict(X_test)

    return predictions
def evaluate_model(y_test, predictions):
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    }
def run_linear_regression():
    X_train, X_test, y_train, y_test = prepare_data(
        "data/raw/housing.csv"
    )

    model = train_linear_regression(X_train, y_train)

    predictions = predict(model, X_test)

    results = evaluate_model(y_test, predictions)

    print("\nLinear Regression Results:")

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
def run_simple_linear_regression():
    X_train, X_test, y_train, y_test = prepare_simple_data(
        "data/raw/housing.csv"
    )

    model = train_simple_linear_regression(X_train, y_train)

    predictions = predict_simple(model, X_test)

    results = evaluate_model(y_test, predictions)

    print("\nSimple Linear Regression Results:")

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    run_linear_regression()
    run_simple_linear_regression()
    run_polynomial_regression()
    run_svr()
    run_decision_tree()
    run_random_forest()
    run_random_forest_tuning()