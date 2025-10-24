import os
import sys
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np
from typing import Optional

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from services.data_pipeline_service import DataPipelineService
from models.model_trainer import FinancialModel

app = FastAPI(
    title="Financial Prediction API",
    description="An API to train a model and get financial predictions. Can use live or local data.",
    version="1.3.0"
)

class TickerRequest(BaseModel):
    ticker: str = "IBM"
    interval: str = "60min"
    window_size: int = 10
    use_local_data: bool = False
    local_data_path: Optional[str] = None

@app.post("/train", tags=["Model Training"])
def train_model_endpoint(request: TickerRequest):
    try:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        pipeline = DataPipelineService(api_key, request.use_local_data, request.local_data_path)
        features, targets, _ = pipeline.run(request.ticker, request.interval, request.window_size)
        
        if features.size == 0:
            raise ValueError("Feature set is empty. Cannot train model.")

        input_shape = (features.shape[1], features.shape[2])
        model_trainer = FinancialModel(ticker=request.ticker)
        model_trainer.build(input_shape)
        model_trainer.train(features, targets)
        model_path = model_trainer.save()
        
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
    window_size: int = Query(10, description="The sequence window size used during training."),
    use_local_data: bool = Query(False, description="Set to true to use local sample data."),
    local_data_path: Optional[str] = Query(None, description="Path to local CSV file.")
):
    try:
        model_trainer = FinancialModel.load(ticker)
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        pipeline = DataPipelineService(api_key, use_local_data, local_data_path)
        features, _, _ = pipeline.run(ticker, interval, window_size)
        
        if features.size == 0:
            raise ValueError("Feature set is empty, cannot make a prediction.")

        last_sequence = features[-1]
        prediction_input = np.expand_dims(last_sequence, axis=0)
        prediction = model_trainer.model.predict(prediction_input)
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
