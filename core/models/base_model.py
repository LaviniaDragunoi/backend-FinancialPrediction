from abc import ABC, abstractmethod
import pandas as pd


class BaseModel(ABC):

    def __init__(self, model_name: str, hyperparameters: dict):
        self.model_name = model_name
        self.hyperparameters = hyperparameters
        self.model = None  # This will hold the trained Keras/PyTorch/SKLearn object

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Trains the model using the provided data."""
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Generates predictions for new data."""
        pass

    @abstractmethod
    def save(self, file_path: str):
        """Saves the trained model state (weights, architecture, etc.)."""
        pass

    @abstractmethod
    def load(self, file_path: str):
        """Loads a model from a saved state."""
        pass