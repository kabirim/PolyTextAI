from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
import asyncio
import uvicorn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))) 

from models.summarizer_model import summairizing
from predict.predict import predict


class InputData(BaseModel):
    gender: int
    age: int
    contract_length: int

class InputText(BaseModel):
    text: str

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": {"theme": "obsidian"}})

@app.post("/predict")
async def get_prediction(data: InputData):
    input_dict = data.model_dump()  
    prediction = predict(input_dict)
    return {"prediction": int(prediction)} # ← conversion ici obligatoire

@app.post("/summerizing")
async def get_summerizing(input: InputText):
    result = asyncio.run(summairizing(input.text))
    return {"Output": result}

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8001)