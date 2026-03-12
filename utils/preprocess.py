import pandas as pd

def preprocess_data(df):
    # Assuming 'Class' is the target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Handle any preprocessing like feature selection, etc.
    # For now, just return X, y
    return X, y