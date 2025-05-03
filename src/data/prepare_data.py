import sys
import os
# On obtient le chemin vers : summerizedText/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))) 
from src.features.preprocessing import encode_features
import load_data as ld

def prepare_data():
    raw_path = '../../data/raw/customers.csv'
    processed_path = 'data/processed/train.csv'
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f'{raw_path} not found')

    df = ld.load_data(raw_path)
    df = encode_features(df)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f'Data prepared and saved to {processed_path}')

# Exécute prepare_data() seulement si ce fichier est exécuté directement, et non pas s’il est importé comme module dans un autre fichier
if __name__ == '__main__':
    prepare_data()
