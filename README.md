# 🧠 NLP API with FastAPI – 6 Powerful Natural Language Processing Features

## 🚀 Project Objective

This project aims to deliver a RESTful API built with FastAPI, offering six core features in Natural Language Processing (NLP). Leveraging state-of-the-art models and deep learning architectures, the API provides advanced text processing capabilities such as summarization, text prediction, auto-correction, entity recognition, and question answering. The objective is to create a modular, scalable, and efficient NLP service that can be easily integrated into other applications.

---

## 📚 Tech Stack

- **Programming Language**: Python
- **Framework**: FastAPI
- **API Server**: Uvicorn
- **Libraries & Tools**:
  - Machine Learning: `TensorFlow`, `Keras`, `PyTorch`, `scikit-learn`
  - NLP: `Transformers`, `spaCy`, `NLTK`, `datasets`
  - Utilities: `Pickle`, `Pandas`, `NumPy`, `Pydantic`, `asyncio`

---

## 🔧 Features & Endpoints

### 1. 🔍 Summarization
- **Model**: `facebook/bart-large-cnn` (Transformers)
- **Description**: Generate a concise summary from a larger body of text using a pretrained BART model.

### 2. ✍️ Next Word Prediction
- **Architecture**: Embedding → LSTM (RNN) → Dense
- **Libraries**: TensorFlow, Keras, Pickle
- **Description**: Predict the next word in a sequence using a trained deep learning model.

### 3. 🛠️ Autocorrector
- **Library**: NLTK
- **Description**: Automatically correct misspelled words using NLP-based edit distance techniques.

### 4. ⌨️ Text Auto Completion
- **Model**: BART (Encoder-Decoder with BERT + GPT)
- **Description**: Complete an unfinished sentence or phrase using pretrained BART architecture.

### 5. 🏷️ Named Entity Recognition (NER)
- **Library**: spaCy (`en_core_web_sm`)
- **Description**: Extract and classify named entities (like people, organizations, places) from text.

### 6. ❓ Question Answering
- **Models**: `bert-base-uncased (fine-tuned on SQuAD 2.0)` and `bert-large-uncased-whole-word-masking-finetuned-squad (pretrained)`
- **Libraries**: Transformers (`BertForQuestionAnswering`, `AutoTokenizer`)
- **Description**: This feature supports two approaches:
Custom fine-tuned model trained on the SQuAD 2.0 dataset, including preprocessing, training, evaluation, and custom query prediction with evaluation metrics.
Pretrained BERT QA model (bert-large-uncased-whole-word-masking-finetuned-squad) used out-of-the-box for general question answering tasks based on a given context.