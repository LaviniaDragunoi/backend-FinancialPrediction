from numpy.f2py.crackfortran import verbose

from core.models.base_model import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMModel(BaseModel):

    def __init__(self, model_name: str, hyperparameters: dict ):
        if hyperparameters is None:
            hyperparameters = {
                'lstm_units': 50,
                'dropout_rate': 0.2,
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        super().__init__(model_name, hyperparameters)
        self.model = None

    def build_model(self, input_shape: tuple):
        model = Sequential([
            LSTM(units=self.hyperparameters['lstm_units'],
                 return_sequences=False,
                 input_shape=input_shape),
            Dropout(self.hyperparameters['dropout_rate']),
            Dense(units=1),
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.hyperparameters['learning_rate'])
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.model = model

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):

        n_features = X_train.shape[1]
        X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, n_features))
        self.build_model((X_train_reshaped.shape[1], n_features))
        self.history = self.model.fit(
            X_train_reshaped,
            y_train.values,
            epochs=self.hyperparameters['epochs'],
            batch_size=self.hyperparameters['batch_size'],
            verbose=2,
            shuffle=False
        )
        print(f"{self.model_name} training complete.")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

        predictions = self.model.predict(X_test_reshaped).flatten()
        return pd.Series(predictions, index=X_test.index)

    def save(self, file_path: str):

        if self.model is None:
            raise ValueError("Model has not been trained; nothing to save.")
        self.model.save(file_path)
        print(f"Model saved to {file_path}.")

    def load(self, file_path: str):
        self.model = load_model(file_path)
        print(f"Model loaded from {file_path}.")