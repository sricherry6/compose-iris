import os
import uvicorn
import requests
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from utils import process_data

TRAINR_ENDPOINT = os.getenv("TRAINR_D_ENDPOINT")

# defining the main app
app = FastAPI(title="processrd", docs_url="/")

# class which is expected in the payload while training
class DataIn(BaseModel):
    number_of_times_pregnent: float
    plasma_glucose: float
    blood_pressure: float
    skinfold_thickness : float
    serum_insulin: float
    bmi:float
    diabetes_pf:float
    age:float
    isdiabetic:str


# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}


@app.post("/process", status_code=200)
# Route to take in data, process it and send it for training.
def process(data: List[DataIn]):
    processed = process_data(data)
    # send the processed data to trainr for training
    response = requests.post(f"{TRAINR_ENDPOINT}/train", json=processed)
    return {"detail": "Processing successful"}


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=8880, reload=True)
