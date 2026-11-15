import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
def create_features(df):
    df = df.copy()

    df["rooms_per_household"] = (
        df["total_rooms"] / df["households"]
    )

    df["bedrooms_per_room"] = (
        df["total_bedrooms"] / df["total_rooms"]
    )

    df["population_per_household"] = (
        df["population"] / df["households"]
    )

    return df
numeric_features = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household"
]

categorical_features = [
    "ocean_proximity"
]
numeric_transformer =Pipeline(steps=[
    ("imputer",
     SimpleImputer(strategy="median")),
     ("scaler",StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
) 
def load_data(file_path):
    df = pd.read_csv(file_path)

    df = create_features(df)

    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]

    return X, y
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test
def preprocess_data(X_train, X_test):
    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed
def prepare_data(file_path):
    X, y = load_data(file_path)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_processed, X_test_processed = preprocess_data(
        X_train, X_test
    )

    return X_train_processed, X_test_processed, y_train, y_test