import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

def load_csv_data(path: str, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    return pd.read_csv(path, sep=sep, encoding=encoding)

def load_and_clean_text_from_file(path: str) -> str:
    with open(path, "r", encoding="utf8") as file:
        data = file.read()

    # Nettoyage de caractères spéciaux
    data = data.replace('\r', '').replace('\ufeff', '')

    # Optionnel : tout en minuscules
    data = data.lower()

    # Découpe le texte en phrases
    # Utilise généralement la ponctuation (comme ., !, ?) pour séparer les phrases.
    # "Hello! How are you? I'm fine." => ['Hello!', 'How are you?', "I'm fine."]
    sentences = sent_tokenize(data)

    detokenizer = TreebankWordDetokenizer()
    cleaned_sentences = []

    for sentence in sentences:
        # Conserve les mots avec apostrophes (ex: don't)
        # Découper une phrase ou un texte en mots et ponctuations
        # ['I', "'m", 'learning', 'NLP', '!']
        tokens = word_tokenize(sentence)
        words = [word for word in tokens if re.match(r"[A-Za-z0-9']+$", word)]
        # Reconstituer une phrase à partir d’une liste de tokens. Utilisé pour retrouver un texte à partir de mots séparés
        # tokens = ['I', "'m", 'learning', 'NLP', '!'] => "I'm learning NLP!"
        cleaned_sentence = detokenizer.detokenize(words)
        cleaned_sentences.append(cleaned_sentence)

    # Recompose le texte avec des points
    final_text = '. '.join(cleaned_sentences) + '.'
    return final_text

def load_txt_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at the location '{path}' is not found")