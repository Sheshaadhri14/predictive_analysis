from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel
from app.model import train_model, predict_downtime
import os

app = FastAPI()

DATA_FILE = "app/data/uploaded_data.csv"
MODEL_FILE = "app/data/model.pkl"

class PredictionInput(BaseModel):
    Temperature: float
    Run_Time: float

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        if not {"Machine_ID", "Temperature", "Run_Time", "Downtime_Flag"}.issubset(df.columns):
            raise HTTPException(status_code=400, detail="Invalid CSV format.")
        df.to_csv(DATA_FILE, index=False)
        return {"message": "File uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/train")
def train_endpoint():
    if not os.path.exists(DATA_FILE):
        raise HTTPException(status_code=400, detail="No data uploaded.")
    try:
        metrics = train_model(DATA_FILE, MODEL_FILE)
        return {"message": "Model trained successfully.", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict")
def predict_endpoint(input_data: PredictionInput):
    try:
        prediction = predict_downtime(MODEL_FILE, input_data.dict())
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
