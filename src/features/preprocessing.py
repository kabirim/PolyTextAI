from sklearn.preprocessing import LabelEncoder

def encode_features(df):
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    return df