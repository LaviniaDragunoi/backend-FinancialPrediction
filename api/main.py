import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from services.data_pipeline_service import run_data_pipeline

app = FastAPI(
    title="Financial Prediction API",
    description="An API to train a model and get financial predictions.",
    version="0.1.0"
)

class TickerRequest(BaseModel):
    ticker: str = "IBM"
    interval: str = "60min"
    window_size: int = 10

@app.post("/train")
def train_model(request: TickerRequest):
    """
    Receives a ticker, runs the data pipeline, and (in the future) trains a model.
    """
    try:
        print(f"Received training request for {request.ticker}")
        features, targets, cleaned_df = run_data_pipeline(request.ticker, request.interval, request.window_size)
        
        # --- Placeholder for Model Training ---
        # Next step will be to take 'features' and 'targets' and train a model.
        # For now, we'll just confirm the data is processed.
        
        # Prepare a summary of the cleaned data for the response
        cleaned_data_summary = cleaned_df.head().to_dict(orient="records")

        return {
            "message": f"Data pipeline completed for {request.ticker}.",
            "features_shape": features.shape,
            "targets_shape": targets.shape,
            "cleaned_data_head": cleaned_data_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict")
def get_prediction():
    """
    (Placeholder) Returns financial predictions from the trained model.
    """
    # --- Placeholder for Prediction ---
    # In the future, this will load the trained model and return real predictions.
    return {"message": "Prediction endpoint is not yet implemented. Train a model first using the POST /train endpoint."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
