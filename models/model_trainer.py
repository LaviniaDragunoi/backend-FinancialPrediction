import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import os

def build_model(input_shape: tuple) -> Sequential:
    """
    Builds a simple LSTM model.
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model: Sequential, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 20, batch_size: int = 32):
    """
    Trains the given model.
    """
    print("Starting model training...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    print("Model training completed.")
    return history

def save_model(model: Sequential, ticker: str) -> str:
    """
    Saves the trained model to a file.
    """
    models_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f'{ticker}_model.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model_path

def load_model(ticker: str) -> Sequential:
    """
    Loads a saved model from a file.

    Args:
        ticker: The ticker symbol for the model to load.

    Returns:
        The loaded Keras model.
    """
    models_dir = os.path.join(os.path.dirname(__file__))
    model_path = os.path.join(models_dir, f'{ticker}_model.keras')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
    print(f"Loading model from {model_path}")
    model = keras_load_model(model_path)
    return model
