import pickle
import pandas as pd
import numpy as np
import sys
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models.named_entity_recognition_model import get_named_entity_recongnition

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))) 
from models.automatic_text_completion_model import generate_automatic_text_completion
from models.questions_answering_bert_squandV1 import give_an_answer
from models.autocorrector_model import calculate_mispelled_word
from models.cleanRawText import clean_raw_text
from models.score_cv_job import compute_similarity, match_years_of_experience, match_exact, match_any_overlap

def predict(input_dict):
    with open('src/models/models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    input_df = pd.DataFrame([input_dict])
    return model.predict(input_df)[0]

def predictNextWord(inputText):
    with open('src/models/savedModels/model_nextWordPrediction.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('src/models/savedModels/tokenizer_next_word.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
     # Generate next word predictions
    seed_text = inputText
    next_words = 1

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
        [token_list], maxlen=142-1, padding='pre')
        predicted_probs = model.predict(token_list)
        predicted_word = tokenizer.index_word[np.argmax(predicted_probs)]
        seed_text += " " + predicted_word

    return seed_text

def predictAnswer(inputObject):
    return give_an_answer(inputObject.context, inputObject.querie,inputObject.answer)

def get_best_correction(word,max_suggestions=3):
     return calculate_mispelled_word(word,max_suggestions)

def predict_automatic_text_completion(sentence):
    return generate_automatic_text_completion(sentence)

def predict_named_entity_recongnition(content):
    return get_named_entity_recongnition(content)

def cleaner_raw_text(inputText):
    return clean_raw_text(inputText)

def score_cv(cv, job):
    scores = {
        "skills": compute_similarity(cv.get("skills", []), job.get("skills", [])),
        "tools": compute_similarity(cv.get("tools", []), job.get("tools", [])),
        "experience": match_years_of_experience(cv.get("experience", 0), job.get("experience", 0)),
        "seniority": match_exact(cv.get("seniority", ""), job.get("seniority", "")),
        "certifications": compute_similarity(cv.get("certifications", []), job.get("certifications", [])),
        "languages": compute_similarity(cv.get("languages", []), job.get("languages", [])),
        "locations": match_any_overlap(cv.get("locations", []), job.get("locations", [])),
    }

    weights = {
        "skills": 0.3,
        "tools": 0.2,
        "experience": 0.2,
        "seniority": 0.1,
        "certifications": 0.05,
        "languages": 0.1,
        "locations": 0.05
    }

    total_score = sum(scores[key] * weights[key] for key in scores)

    return {
        "score": round(total_score, 2),
        "details": scores
    }