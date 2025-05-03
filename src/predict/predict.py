import pickle
import pandas as pd

def predict(input_dict):
    with open('src/models/models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    input_df = pd.DataFrame([input_dict])
    input_df
    return model.predict(input_df)[0]
