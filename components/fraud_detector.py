import numpy as np
import joblib

model = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/scaler.pkl")

def predict_transaction(features):

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]

    probability = model.predict_proba(scaled)[0][1]

    return prediction, probability