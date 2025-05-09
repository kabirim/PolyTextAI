import nltk
import re
import string
import sys
import os
from nltk.stem import WordNetLemmatizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))) 
import load_data as ld

def get_words_from_vocabulary():
    file_path = 'src/data/data/processed/vocabulary.txt'
    words = ld.loadfileTxt(file_path)
    words = [word.strip() for word in words.split(',')]
    vocab = set(words)
    return words, vocab

words,vocab = get_words_from_vocabulary()

# To calculate the probability of the correct word prediction, we compute how often each word appears in the dataset
# Words that appear more frequently are likely to be correct and the frequency count of each word is stored in a dictionary
def count_word_frequency(words):
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    return word_count

word_count = count_word_frequency(words)

# Using the word frequency data, we calculate the probability of each word
# Words that appear more often will have a higher probability
def calculate_probability(word_count):
    total_words = sum(word_count.values())
    return {word: count / total_words for word, count in word_count.items()}

probabilities = calculate_probability(word_count)

lemmatizer = WordNetLemmatizer()

# Is used to lemmatize a word, that is, to reduce it to its canonical or base form, using the WordNet lemmatizer provided by NLTK
# 'running' => 'run', 'cars' => 'car'
def lemmatize_word(word):
    """Lemmatize a given word using NLTK WordNet Lemmatizer."""
    return lemmatizer.lemmatize(word)

# Deleting a letter: removes a letter from the word
# On construit une nouvelle version du mot sans le caractère à la position 
def delete_letter(word):
    return [word[:i] + word[i+1:] for i in range(len(word))]

# Swapping adjacent letters: swaps adjacent letters in the word
# Pour chaque i, on échange les lettres à la position i et i+1
def swap_letters(word):
    return [word[:i] + word[i+1] + word[i] + word[i+2:] for i in range(len(word)-1)]

# Replacing a letter: replaces each letter with every other letter of the alphabet
def replace_letter(word):
    letters = string.ascii_lowercase
    return [word[:i] + l + word[i+1:] for i in range(len(word)) for l in letters]

# Inserting a new letter: inserts a new letter at every position in the word
def insert_letter(word):
    letters = string.ascii_lowercase
    return [word[:i] + l + word[i:] for i in range(len(word)+1) for l in letters]

def generate_corrections(word):
    corrections = set()
    corrections.update(delete_letter(word))
    corrections.update(swap_letters(word))
    corrections.update(replace_letter(word))
    corrections.update(insert_letter(word))
    return corrections

def generate_corrections_level2(word):
    level1 = generate_corrections(word)
    level2 = set()
    for w in level1:
        level2.update(generate_corrections(w))
    return level2

def get_best_correction(word, probs, vocab, max_suggestions=3):
    # Si word est déjà dans le vocabulaire, il est correct → on retourne [word]
    # Sinon :  On génère des corrections de premier niveau avec generate_corrections(word) (par ex. suppression d’une lettre, changement d’une lettre, etc.).
    # On filtre celles qui sont dans le vocabulaire : .intersection(vocab)
    # Si aucune n’est trouvée, on passe à un niveau 2 de corrections avec generate_corrections_level2 (plus éloignées, ex : deux lettres modifiées).
    corrections = (
        [word] if word in vocab else list(generate_corrections(word).intersection(vocab)) or 
        list(generate_corrections_level2(word).intersection(vocab))
    )
    # [("apple", 0.8), ("apply", 0.1), ("app", 0.05)]
    return sorted([(w, probs.get(w, 0)) for w in corrections], key=lambda x: x[1], reverse=True)[:max_suggestions]

if __name__ == '__main__':
    get_words_from_vocabulary()

    user_input = "teste"
    suggestions = get_best_correction(user_input, probabilities, vocab, max_suggestions=3)

    print("\n Top suggestions:")
    for suggestion in suggestions:
        print(suggestion[0])