# Models/RF.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_random_forest(model, X_test, y_test=None):
    preds = model.predict(X_test)
    results = {"predictions": preds}

    if y_test is not None:
        results["mae"] = mean_absolute_error(y_test, preds)
        results["mse"] = mean_squared_error(y_test, preds)
        results["r2"] = r2_score(y_test, preds)

    return results
