import pickle
import pandas as pd
import numpy as np
import sys
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))) 
from models.questions_answering_bert_squandV1 import give_an_answer

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