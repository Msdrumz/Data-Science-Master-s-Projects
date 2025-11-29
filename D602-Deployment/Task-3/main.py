#!/usr/bin/env python
# coding: utf-8

# import statements
from fastapi import FastAPI, HTTPException, Query
import json
import numpy as np
import pickle
import sklearn
import datetime
import uvicorn
import pydantic
from pydantic import BaseModel , ValidationError, constr
from sklearn.preprocessing import PolynomialFeatures



with open("airport_encodings.json", "r") as enc_file:
    airports = json.load(enc_file)

 

def create_airport_encoding(airport: str, airports: dict) -> np.array:
    """
    create_airport_encoding is a function that creates an array the length of all arrival airports from the chosen
    departure aiport.  The array consists of all zeros except for the specified arrival airport, which is a 1.  

    Parameters
    ----------
    airport : str
        The specified arrival airport code as a string
    airports: dict
        A dictionary containing all of the arrival airport codes served from the chosen departure airport
        
    Returns
    -------
    np.array
        A NumPy array the length of the number of arrival airports.  All zeros except for a single 1 
        denoting the arrival airport.  Returns None if arrival airport is not found in the input list.
        This is a one-hot encoded airport array.

    """
    airport = airport.upper()
    temp = np.zeros(len(airports))
    if airport not in airports:
        return None
    temp = np.zeros(len(airports))
    temp[airports.get(airport)] = 1
    return temp
    
 
from datetime import datetime

def time_to_seconds(time_val) -> int:

    # Convert to string in case it's an int
    time_str = str(time_val)
    t = datetime.strptime(time_str.zfill(4), "%H%M")

    return t.hour * 3600 + t.minute * 60

with open("airport_encodings.json", "r") as enc_file:
    airports = json.load(enc_file)

with open("finalized_model.pkl", "rb") as file:
    model = pickle.load(file)

poly = PolynomialFeatures(degree=1)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is functional"}


@app.get("/predict/delays")
def predict_delay(
    arrival_airport: str = Query(..., min_length=3, max_length=3, description="Arrival airport IATA code (e.g. PHL)"),
    local_departure_time: str = Query(..., description="Local departure time in HHMM or HMM (24-hour)"),
    local_arrival_time: str = Query(..., description="Local arrival time in HHMM OR HMM (24-hour)")
):
    # 1. Encode airport
    if arrival_airport not in airports:
        raise HTTPException(status_code=400, detail="Unknown airport code")
    encoded_airport = create_airport_encoding(arrival_airport, airports)
    
    
    # 2. Convert times 
    try:
        dep_sec = time_to_seconds(local_departure_time)
        arr_sec = time_to_seconds(local_arrival_time)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time format. Use HHMM or HMM.")

    # 3. Build feature vector
    input_data = np.concatenate(([1], encoded_airport, [dep_sec], [arr_sec]))
    input_data = input_data.reshape(1, -1)
    

    # Predict
    try:
        pred = model.predict(input_data)
        delay = float(pred[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"average_departure_delay_minutes": delay}
   

    # 5. Return JSON
    return {"average_departure_delay_minutes": float(delay[0])}

    


