from typing import Annotated, List, Optional
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

PyObjectId = Annotated[str, BeforeValidator(str)]

class HouseModel(BaseModel):
   
    id: Optional[PyObjectId] = Field(alias="_id", default=None) 
    Bedrooms: int= Field(...)
    bathrooms: int= Field(...)
    Living_m2: str= Field(...)
    Lot_m2: str= Field(...)
    Grade: int= Field(...)
    Lat: str= Field(...)
    Long: str= Field(...)
    Price_Predicted: float= Field(...)
    Selected_model: str= Field(...)
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "Bedrooms": "3",
                "bathrooms": "1",
                "Living_m2": "109",
                "Lot_m2": "500",
                "Grade": "7",
                "Zipcode": "98178",
                "Lat": "47.5112",
                "Long": "-122.257",
                "Price_Predicted": 100.00
            }
        },
    )




class HouseCollection(BaseModel):
    predictions: List[HouseModel]
