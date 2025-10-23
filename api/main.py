import os
import sys
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from services.data_pipeline_service import run_data_pipeline
from models.model_trainer import build_model, train_model, save_model, load_model

app = FastAPI(
    title="Financial Prediction API",
    description="An API to train a model and get financial predictions.",
    version="1.0.0"  # Updated version to 1.0.0
)

class TickerRequest(BaseModel):
    ticker: str = "IBM"
    interval: str = "60min"
    window_size: int = 10

@app.post("/train", tags=["Model Training"])
def train_model_endpoint(request: TickerRequest):
    """
    Receives a ticker, runs the data pipeline, and trains a predictive model.
    """
    try:
        # 1. Run the data pipeline
        print(f"Received training request for {request.ticker}")
        features, targets, _ = run_data_pipeline(request.ticker, request.interval, request.window_size)
        
        if features.size == 0:
            raise ValueError("Feature set is empty. Cannot train model. Check data source or window size.")

        # 2. Build the model
        input_shape = (features.shape[1], features.shape[2])
        model = build_model(input_shape)
        
        # 3. Train the model
        train_model(model, features, targets)
        
        # 4. Save the model
        model_path = save_model(model, request.ticker)
        
        return {
            "message": f"Model for {request.ticker} trained and saved successfully.",
            "model_path": model_path
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict", tags=["Prediction"])
def get_prediction(
    ticker: str = Query("IBM", description="The stock ticker symbol to predict."),
    interval: str = Query("60min", description="The time interval for the data."),
    window_size: int = Query(10, description="The sequence window size used during training.")
):
    """
    Loads a trained model and returns a prediction for the next time step.
    """
    try:
        # 1. Load the trained model
        model = load_model(ticker)

        # 2. Get the latest data sequence
        print(f"Fetching latest data for {ticker} to make a prediction...")
        features, _, _ = run_data_pipeline(ticker, interval, window_size)
        
        if features.size == 0:
            raise ValueError("Feature set is empty, cannot make a prediction.")

        # 3. Prepare the last sequence for prediction
        last_sequence = features[-1]
        prediction_input = np.expand_dims(last_sequence, axis=0) # Reshape to (1, window_size, num_features)

        # 4. Make the prediction
        prediction = model.predict(prediction_input)
        predicted_value = float(prediction[0][0])

        return {
            "ticker": ticker,
            "prediction_for_next_step": predicted_value
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
