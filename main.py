# IMPORT the libraries
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

#load the model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('cv.pkl')

#define the request body
class Emailrequst(BaseModel):
    email:str

#define the home endpoint   
app = FastAPI()
@app.get("/")
def home():
    return{"message":"welcome to the email classifier API!"}


# define the predict endpoint
@app.post("/predict")
def predict(request: Emailrequst):

    email = request.email
    eamil_vectorized = vectorizer.transform([email])
    prediction = model.predict(eamil_vectorized)
    return {"prediction": prediction[0]}