import json
from typing import Dict, List
from bson import ObjectId
from discord import Status
from fastapi import Body, FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from fastapi import FastAPI, Body, HTTPException, status
from fastapi.responses import Response
import joblib
from matplotlib import pyplot as plt
import numpy as np
from pydantic import BaseModel
import motor.motor_asyncio
from Models.House import HouseCollection, HouseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Predict Laptop price API",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


client = motor.motor_asyncio.AsyncIOMotorClient("mongodb+srv://EN_IA:sZMHrjGr2nX7XglQ@cluster0.qd9iypu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.houses
prediction_collection = db.get_collection("predictions")

@app.post(
    "/prediction/create",
    response_description="Add new prediction",
    response_model=HouseModel,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_prediction(prediction: HouseModel = Body(...)):
   
    house ={"data":[[   prediction.Bedrooms,
                        prediction.bathrooms,
                        prediction.Living_m2,
                        prediction.Lot_m2,
                        prediction.Grade,
                        prediction.Lat,
                        prediction.Long
                        
                    ]]}
    
    house_json = json.dumps(house)
    predicted_price =await predictPrice(house_json,prediction.Selected_model)

    prediction.Price_Predicted = predicted_price[0]
    
    new_predicion = await prediction_collection.insert_one(
        prediction.model_dump(by_alias=True, exclude=["id"])
    )
    created_prediction = await prediction_collection.find_one(
        {"_id": new_predicion.inserted_id}
    )
    return created_prediction


@app.get(
    "/prediction/list",
    response_description="List all predictions",
    response_model=HouseCollection,
    response_model_by_alias=False,
)
async def list_predictions_endpoint():
  return HouseCollection(predictions=await prediction_collection.find().to_list(1000))


@app.get(
    "/prediction/{id}",
    response_description="Get a prediction",
    response_model=HouseModel,
    response_model_by_alias=False,
)
async def show_prediction(id):
    """
    Get the prediction with the same id
    """
    prediction = await prediction_collection.find_one({"_id": ObjectId(id)})
    if (
       prediction
    ) is not None:
        return prediction
    raise HTTPException(status_code=404, detail=f"Prediction {id} not found")

                    
@app.delete("/prediction/delete/{id}", response_description="Delete a prediction")
async def delete_prediction(id):
    """
    Remove a prediction  
    """
    delete_result = await prediction_collection.delete_one({"_id": ObjectId(id)})
    if delete_result.deleted_count == 1:
        return Response("Deleted")
    raise HTTPException(status_code=404, detail=f"Prediction {id} not found")


@app.get("/models", response_model=Dict[str, List[Dict[str, str]]])
async def get_models():
    return {"models": [{"name": "RandomForestRegressor(Default)"},
                       {"name": "DecisionTreeRegressor(Default)"},
                       {"name": "GradientBoostingRegressor(Default)"},
                       {"name": "RandomForestRegressor(Tuning)"},
                       {"name": "DecisionTreeRegressor(Tuning)"},
                       {"name": "GradientBoostingRegressor(Tuning)"},
                       ]}


async def predictPrice(obj,selected_model):

    if selected_model == "RandomForestRegressor(Default)":
        model = joblib.load("modelo_rf.zip")
    elif selected_model == "DecisionTreeRegressor(Default)":
        model = joblib.load("modelo_dt.zip")
    elif selected_model == "GradientBoostingRegressor(Default)":
        model = joblib.load("modelo_gb.zip")
    elif selected_model == "RandomForestRegressor(Tuning)":
        model = joblib.load("modelo_rf_tuning.zip")
    elif selected_model == "DecisionTreeRegressor(Tuning)":
        model = joblib.load("modelo_dt_tuning.zip")
    elif selected_model == "GradientBoostingRegressor(Tuning)":
        model = joblib.load("modelo_gb_tuning.zip")

    input_data = json.loads(obj)['data']
    input_array = np.array(input_data)
    prediction = model.predict(input_array)
   
    print(prediction)
    return prediction



