"""
Saves the model to the specified path
"""
import joblib
import os


def save_model(model, path):
    # Save the model to the specified path
    joblib.dump(model, path)
    print(f'Model saved to {path}')
