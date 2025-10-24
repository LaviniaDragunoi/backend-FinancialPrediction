import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import os
from typing import List, Optional

class FinancialModel:
    """
    A class to handle the building, training, saving, and loading of a financial prediction model.
    """
    def __init__(self, model: Optional[Sequential] = None, ticker: str = "default"):
        """
        Initializes the FinancialModel.

        Args:
            model: A pre-existing Keras model. If None, a new model is built.
            ticker: The ticker symbol for the model.
        """
        self.model = model
        self.ticker = ticker
        self.history = None

    def build(self, input_shape: tuple, lstm_units: List[int] = [50, 50], dropout_rate: float = 0.2, dense_units: int = 25):
        """
        Builds a Keras LSTM model.

        Args:
            input_shape: The shape of the input data.
            lstm_units: A list of integers for the number of units in each LSTM layer.
            dropout_rate: The dropout rate.
            dense_units: The number of units in the Dense layer.
        """
        model = Sequential()
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            if i == 0:
                model.add(LSTM(units=units, return_sequences=return_sequences, input_shape=input_shape))
            else:
                model.add(LSTM(units=units, return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))

        model.add(Dense(units=dense_units))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 20, batch_size: int = 32):
        """
        Trains the model.

        Args:
            X_train: The training data.
            y_train: The training labels.
            epochs: The number of epochs to train for.
            batch_size: The batch size.
        """
        if self.model is None:
            raise ValueError("Model has not been built or loaded. Call build() or load() first.")

        print("Starting model training...")
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
        print("Model training completed.")

    def save(self, models_dir: Optional[str] = None) -> str:
        """
        Saves the trained model to a file.

        Args:
            models_dir: The directory to save the model in. Defaults to the 'models' directory.

        Returns:
            The path to the saved model.
        """
        if self.model is None:
            raise ValueError("No model to save.")

        if models_dir is None:
            models_dir = os.path.dirname(__file__)

        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, f'{self.ticker}_model.keras')
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        return model_path

    @classmethod
    def load(cls, ticker: str, models_dir: Optional[str] = None) -> 'FinancialModel':
        """
        Loads a saved model from a file.

        Args:
            ticker: The ticker symbol for the model to load.
            models_dir: The directory where the model is saved. Defaults to the 'models' directory.

        Returns:
            A FinancialModel instance with the loaded model.
        """
        if models_dir is None:
            models_dir = os.path.dirname(__file__)

        model_path = os.path.join(models_dir, f'{ticker}_model.keras')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

        print(f"Loading model from {model_path}")
        model = keras_load_model(model_path)
        return cls(model=model, ticker=ticker)

    def get_summary(self):
        """Prints the model summary."""
        if self.model:
            self.model.summary()
        else:
            print("Model not built or loaded yet.")

