from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

import uvicorn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))) 

from predict.predict import predict

class InputData(BaseModel):
    gender: int
    age: int
    contract_length: int

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": {"theme": "obsidian"}})

@app.post("/predict")
async def get_prediction(data: InputData):
    input_dict = data.model_dump()  
    prediction = predict(input_dict)  # ← généralement un np.int64
    return {"prediction": int(prediction)}    # ← conversion ici obligatoire

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8001)