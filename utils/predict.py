import numpy as np

def predict_fraud(data, model, scaler):
    # Scale the data
    data_scaled = scaler.transform(data)
    
    # Predict
    predictions = model.predict(data_scaled)
    
    return predictions