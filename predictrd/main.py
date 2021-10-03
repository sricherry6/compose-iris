import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from utils import load_model, predict

# defining the main app
app = FastAPI(title="predictrd", docs_url="/")

# class which is expected in the payload
class QueryIn(BaseModel):
    number_of_times_pregnent: float
    plasma_glucose: float
    blood_pressure: float
    skinfold_thickness : float
    serum_insulin: float
    bmi:float
    diabetes_pf:float
    age:float


# class which is returned in the response
class QueryOut(BaseModel):
    is_diabetic: str


# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}


@app.post("/predict_diabetes", response_model=QueryOut, status_code=200)
# Route to do the prediction using the ML model defined.
# Payload: QueryIn containing the parameters
# Response: QueryOut containing the flower_class predicted (200)
def predict_diabetes(query_data: QueryIn):
    output = {"is_diabetic": predict(query_data)}
    return output


@app.post("/reload_model", status_code=200)
# Route to reload the model from file
def reload_model():
    load_model()
    output = {"detail": "Model successfully loaded"}
    return output


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=9990, reload=True)
