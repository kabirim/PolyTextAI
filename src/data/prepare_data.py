import sys
import os
import re

# On obtient le chemin vers : summerizedText/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))) 
from src.features.preprocessing import encode_features
import load_data as ld

def prepare_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # chemin du script actuel
    raw_path = 'data/raw/customers.csv'
    processed_path = os.path.join(base_dir, 'data', 'processed', 'train.csv')
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f'{raw_path} not found')

    df = ld.loadCsv_data(raw_path)
    df = encode_features(df)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f'Data prepared and saved to {processed_path}')

def read_text_data():
    raw_path = 'data/raw/metamorphosis_clean.txt'

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f'{raw_path} not found')
    
    df = ld.loadTxt_data_and_clean(raw_path)
    print(f'Data prepared and saved to {df}')

def extract_all_words_sequences():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # chemin du script actuel
    file_path = 'data/raw/vocabulary.txt'
    processed_path = os.path.join(base_dir, 'data', 'processed', 'vocabulary.txt')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'{file_path} not found')
    
    df = ld.loadfileTxt(file_path)
    # Extraction des mots
    words = re.findall(r'\w+', df)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    # Écriture des mots dans le fichier, un par ligne
    with open(processed_path, 'w', encoding='utf-8') as f:
        for word in words:
            f.write(word + ',')
    print(f'Data prepared and saved to {processed_path}')

# Exécute prepare_data() seulement si ce fichier est exécuté directement, et non pas s’il est importé comme module dans un autre fichier
if __name__ == '__main__':
    extract_all_words_sequences()
