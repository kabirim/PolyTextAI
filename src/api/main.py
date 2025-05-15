from fastapi import FastAPI
from pydantic import BaseModel,Field
import sys
import os
from typing import List
import asyncio
import uvicorn
from fastapi import status
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))) 

from models.summarizer_model import summairizing
from predict.predict import predict, predict_named_entity_recongnition,predictNextWord,predictAnswer,get_best_correction,predict_automatic_text_completion

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

@app.post("/predict/churn", status_code=status.HTTP_200_OK)
async def predict_churn(data: InputData):
    input_dict = data.model_dump()  
    prediction = predict(input_dict)
    return {"prediction": int(prediction)}

@app.post("/text/summary",status_code=status.HTTP_200_OK )
async def summarize_text(input: InputText):
    result = asyncio.run(summairizing(input.text))
    return {"Output": result}

@app.post("/predict/next-word",status_code=status.HTTP_200_OK)
async def predict_next_word(input: InputText):
    result = predictNextWord(input.text)
    return {"Output": result}

@app.post("/qa/predict",status_code=status.HTTP_200_OK)
async def predict_answer(input: InputQA):
    result = predictAnswer(input)
    return {"Output": result}

@app.post("/text/autocorrect",status_code=status.HTTP_200_OK)
async def autocorrect_text(input: InputText):
    result = get_best_correction(input.text,3)
    return {"Output": result}

@app.post("/text/completion",status_code=status.HTTP_200_OK)
async def complete_text(input: InputText):
    result = predict_automatic_text_completion(input.text)
    return {"Output": result}

@app.post("/text/ner",status_code=status.HTTP_200_OK)
async def named_entity_recognition(input: InputText):
    result = predict_named_entity_recongnition(input.text)
    return {"Output": result}

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8001)