import uvicorn
import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from utils import init_model, train_model
from typing import List


PREDICTR_ENDPOINT = os.getenv("PREDICTR_D_ENDPOINT")

# defining the main app for Diabetes dataset
app = FastAPI(title="trainrd", docs_url="/")

# calling the load_model during startup.
# this will train the model and keep it loaded for prediction.
app.add_event_handler("startup", init_model)

# class which is expected in the payload while training
class TrainIn(BaseModel):
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


@app.post("/train", status_code=200)
# Route to further train the model based on user input in form of feedback loop
# Payload: FeedbackIn containing the parameters and correct diabetic class
# Response: Dict with detail confirming success (200)
def train(data: List[TrainIn]):
    train_model(data)
    # tell predictr to reload the model
    response = requests.post(f"{PREDICTR_ENDPOINT}/reload_model")
    return {"detail": "Diabetes Training successful"}


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=7770, reload=True)
