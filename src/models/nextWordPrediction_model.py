import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout,GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import regex as re
import sys
import os
import asyncio
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# On obtient le chemin vers : summerizedText/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))) 
import load_data as ld


async def nextWordPrediction(text):
  sentences = [text.strip() for text in re.split(
        r'(?<=[.!?])\s+', text) if text.strip()]
  # Il conserve uniquement les 5000 mots les plus fréquents dans ton texte.
  # Tous les mots rares (au-delà des 5000 premiers) seront ignorés lors de la vectorisation.

  # L’argument oov_token="<OOV>" permet de remplacer les mots rares par un token spécial "Out Of Vocabulary", utile pour ne pas perdre l’information complètement.
  # num_words=2000, oov_token="<OOV>"
  tokenizer = Tokenizer(lower=True)
  tokenizer.fit_on_texts(sentences)
  total_words = len(tokenizer.word_index) + 1 
  with open('src/models/savedModels/tokenizer_next_word.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

  input_sequences = []
  for line in sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
       n_gram_sequence = token_list[:i+1]
       input_sequences.append(n_gram_sequence)
  
  max_sequence_len = max([len(seq) for seq in input_sequences])
  input_sequences = np.array(pad_sequences(
    input_sequences, maxlen=max_sequence_len, padding='pre'))
  X, y = input_sequences[:, :-1], input_sequences[:, -1]
  # Convert target data to one-hot encoding
  # to_categorical transforme chaque entier en un vecteur binaire de taille total_words, où seule la case correspondant à l'indice est à 1..
  # one-hot categorical
  y = tf.keras.utils.to_categorical(y, num_classes=total_words)
  model = await model_creation(total_words,max_sequence_len,X,y)
  await Test(tokenizer,model,max_sequence_len)

async def model_creation(total_words,max_sequence_len,X,y):
    # Define the model
    model = Sequential()
    # Le nombre de mots a gérér par model input_dim
    model.add(Embedding(input_dim = total_words, output_dim = 10, input_length=max_sequence_len-1))
    # Aide à limiter les poids trop grands => kernel_regularizer
    # LSTM plus petit (64 unités) → diminue la complexité.
    model.add(LSTM(units=32,activation="tanh",return_sequences=True))
    model.add(LSTM(32, activation="tanh", return_sequences=False))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units= total_words, activation='softmax'))  # Elle transforme les logits (valeurs brutes) de la dernière couche en probabilités sur toutes les classes  entre 0 et 1 

    # model = Sequential()
    # model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
    # model.add(LSTM(64, return_sequences=True))
    # model.add(LSTM(32))
    # model.add(Dense(128, activation="relu"))
    # model.add(Dense(total_words, activation="softmax")) 

    model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #  évite un entraînement trop long quand la validation n’améliore plus.
    # es = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True) => "callbacks=[es],
    model.fit(X_train, y_train, epochs=300,batch_size=64, validation_data =(X_test,y_test),  verbose=2)

    os.makedirs('src/models/savedModels/models', exist_ok=True)
    with open('src/models/savedModels/model_nextWordPrediction.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    y_pred = model.predict(X_test)
    # y_pred = [
    # [0.1, 0.05, 0.8, 0.05],   # prédiction 1 → classe 2 (index 2)
    # [0.03, 0.9, 0.05, 0.02]   # prédiction 2 → classe 1 (index 1)]
    # y_pred_classes = [2, 1]
    
    # y_test = [
    # [0, 0, 1, 0],   # vraie classe : 2
    # [0, 1, 0, 0]    # vraie classe : 1]
    # y_test_classes = [2, 1]

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    y_pred.shape 
    y_test.shape
    acc = accuracy_score(y_test_classes, y_pred_classes)
    print(f'Model accuracy: {acc:.2f}')
    return model

async def Test(tokenizer,model,max_sequence_len):
    # Generate next word predictions
    seed_text = "machine learning is "
    next_words = 5

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
        [token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list)
        predicted_word = tokenizer.index_word[np.argmax(predicted_probs)]
        seed_text += " " + predicted_word

    print("Next predicted words:", seed_text)

if __name__ == '__main__':
  text = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.
The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.
AI is continuously evolving to benefit many different industries. Machines are wired using a cross-disciplinary approach based on mathematics, computer science, linguistics, psychology, and more.

Machine learning is a subset of AI that focuses on building systems that learn or improve performance based on the data they consume.
Machine learning applications are used in email filtering, speech recognition, computer vision, and more.
Machine learning algorithms are trained to find patterns and features in massive amounts of data in order to make decisions and predictions based on new data.
The better the algorithm, and the more data it has, the more accurate it becomes.

There are three types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
Supervised learning uses labeled data to train algorithms. For example, an input might be an image of an animal, and the label might be "cat".
Unsupervised learning does not use labeled data. Instead, the algorithm must find patterns and relationships in the dataset on its own.
Reinforcement learning is based on the reward system. The algorithm learns to perform a task simply by trying to maximize rewards it receives for its actions.

Deep learning is a subset of machine learning that uses neural networks with many layers.
Neural networks are designed to simulate the behavior of the human brain, allowing machines to recognize patterns and solve complex problems.
Deep learning models require a large amount of data and computational power but have achieved state-of-the-art performance in tasks like image classification, language translation, and game playing.

Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through language.
NLP helps machines understand, interpret, and respond to human language in a valuable way.
Common applications include chatbots, translation services, sentiment analysis, and speech recognition systems.

As AI technologies become more integrated into our daily lives, ethical considerations must be addressed.
Issues like data privacy, algorithmic bias, and transparency are critical to ensuring that AI is developed and used responsibly.
Education and regulation play a crucial role in guiding the development of AI in a direction that benefits society as a whole.

In the future, AI is expected to transform various fields such as healthcare, education, finance, and transportation.
AI-driven innovations can lead to more efficient processes, cost reductions, and new capabilities.
However, it also poses challenges, such as potential job displacement and the need for re-skilling the workforce.
By combining human insight with machine efficiency, the future of AI holds great promise and responsibility.
"""
text = ld.loadTxtData('data/raw/metamorphosis_clean.txt')
asyncio.run(nextWordPrediction(text))