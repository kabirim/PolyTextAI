import pandas as pd
import string
import re
from nltk.tokenize import word_tokenize,sent_tokenize
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer

def loadCsv_data(path):
    return pd.read_csv(path)

def loadTxtData(path):
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
        #  Reconstituer une phrase à partir d’une liste de tokens. Utilisé pour retrouver un texte à partir de mots séparés
        # tokens = ['I', "'m", 'learning', 'NLP', '!'] => "I'm learning NLP!"
        cleaned_sentence = detokenizer.detokenize(words)
        cleaned_sentences.append(cleaned_sentence)

    # Recompose le texte avec des points
    final_text = '. '.join(cleaned_sentences) + '.'
    print(final_text)
    return final_text