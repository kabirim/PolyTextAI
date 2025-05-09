from fastapi import FastAPI
from pydantic import BaseModel,Field
import sys
import os
from typing import List
import asyncio
import uvicorn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))) 

from models.summarizer_model import summairizing
from predict.predict import predict,predictNextWord,predictAnswer

class InputData(BaseModel):
    gender: int
    age: int
    contract_length: int

class InputText(BaseModel):
    text: str

class InputQA(BaseModel):
    context: str = Field(..., description="Le texte dans lequel on cherche la réponse")
    querie: str = Field(..., description="Le text de question à poser")
    answer: str = Field(..., description="Le text de réponse correspondant à la question")

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": {"theme": "obsidian"}})

@app.post("/predict")
async def get_prediction(data: InputData):
    input_dict = data.model_dump()  
    prediction = predict(input_dict)
    return {"prediction": int(prediction)}

@app.post("/summerizing")
async def get_summerizing(input: InputText):
    result = asyncio.run(summairizing(input.text))
    return {"Output": result}

@app.post("/nextWordPrediction")
async def get_nextWordPrediction(input: InputText):
    result = predictNextWord(input.text)
    return {"Output": result}

@app.post("/answerPredicition")
async def get_answerPredicition(input: InputQA):
    result = predictAnswer(input)
    return {"Output": result}

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8001)