from fastapi import FastAPI
from pydantic import BaseModel,Field
import sys
import os
from typing import List
import asyncio
import uvicorn
from fastapi import status
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))) 

from models.summarizer_model import summairizing
from predict.predict import score_cv,cleaner_raw_text, predict, predict_named_entity_recongnition,predictNextWord,predictAnswer,get_best_correction,predict_automatic_text_completion

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

## Score CV    
class CV(BaseModel):
    skills: Optional[List[str]] = []
    tools: Optional[List[str]] = []
    experience: int = 0
    seniority: Optional[str] = ""
    certifications: Optional[List[str]] = []
    languages: Optional[List[str]] = []
    locations: Optional[List[str]] = []

class JobOffer(BaseModel):
    skills: Optional[List[str]] = []
    tools: Optional[List[str]] = []
    experience: int = 0
    seniority: Optional[str] = ""
    certifications: Optional[List[str]] = []
    languages: Optional[List[str]] = []
    locations: Optional[List[str]] = []

class ScoringRequest(BaseModel):
    cv: CV
    job: JobOffer

##End Score

app = FastAPI(
    title="PolyTextAI",
    docs_url="/swagger",          # Swagger UI
    redoc_url="/redoc",           # ReDoc UI
    openapi_url="/openapi.json",  # OpenAPI schema
    swagger_ui_parameters={"syntaxHighlight": {"theme": "obsidian"}}
)

# @app.post("/predict/churn", status_code=status.HTTP_200_OK)
# async def predict_churn(data: InputData):
#     input_dict = data.model_dump()  
#     prediction = predict(input_dict)
#     return {"prediction": int(prediction)}

@app.post("/api/text/clean",status_code=status.HTTP_200_OK )
async def clean_text(input: InputText):
    result = cleaner_raw_text(input.text)
    return result

@app.post("/api/text/summary",status_code=status.HTTP_200_OK )
async def summarize_text(input: InputText):
    result = asyncio.run(summairizing(input.text))
    return {"Output": result}

@app.post("/api/predict/next-word",status_code=status.HTTP_200_OK)
async def predict_next_word(input: InputText):
    result = predictNextWord(input.text)
    return {"Output": result}

@app.post("/api/qa/predict",status_code=status.HTTP_200_OK)
async def predict_answer(input: InputQA):
    result = predictAnswer(input)
    return {"Output": result}

@app.post("/api/text/autocorrect",status_code=status.HTTP_200_OK)
async def autocorrect_text(input: InputText):
    result = get_best_correction(input.text,3)
    return {"Output": result}

@app.post("/api/text/completion",status_code=status.HTTP_200_OK)
async def complete_text(input: InputText):
    result = predict_automatic_text_completion(input.text)
    return {"Output": result}

@app.post("/api/text/ner",status_code=status.HTTP_200_OK)
async def named_entity_recognition(input: InputText):
    result = predict_named_entity_recongnition(input.text)
    return {"Output": result}

@app.post("/api/score",status_code=status.HTTP_200_OK)
async def score_cv_job(request: ScoringRequest):
    result = score_cv(request.cv.model_dump(), request.job.model_dump())
    return result

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8001)