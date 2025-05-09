import time
import string, re
from transformers import BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AdamW

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

dataset = load_dataset("squad_v2")

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# Accéder aux ensembles d'entraînement et de validation
train_data = dataset["train"]
val_data = dataset["validation"]

def gpuAvailable():
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

def load_squad_train_data():
    texts = []
    questions = []
    answers = []
    for q in train_data:
        texts.append(q['context'])
        questions.append(q['question'])
        answers.append(q['answers']) # Dictionnaire avec 'text' et 'answer_start'

    return texts, questions, answers

# Preprocess the data to find answer start and end positions
train_texts, train_queries, train_answers = load_squad_train_data()

def load_squad_validation_data():
    texts = []
    questions = []
    answers = []
    for q in val_data:
       texts.append(q['context'])
       questions.append(q['question'])
       answers.append(q['answers']) # Dictionnaire avec 'text' et 'answer_start'

    return texts, questions, answers

# Preprocess the data to find answer start and end positions
val_texts, val_queries, val_answers = load_squad_validation_data()

# Find the start and end position character
def find_start_end_train_answer_position():
    for (answer, text) in zip(train_answers,train_texts):
        real_answer = answer['text']
        start_index = answer['answer_start']
         # Get the real end index
        end_index = start_index + len(real_answer)

        # S’assurer que la chaîne real_answer correspond bien à un extrait du text (le contexte), en ajustant l’index si besoin.
        if text[start_index:end_index] == real_answer:
            answer['answer_end'] = end_index
        elif text[start_index-1:end_index-1] == real_answer:
            answer['answer_start'] = start_index - 1
            answer['answer_end'] = end_index - 1
        elif text[start_index-2:end_index-2] == real_answer:
            answer['answer_start'] = start_index - 2
            answer['answer_end'] = end_index - 2

find_start_end_train_answer_position()

def find_start_end_validation_answer_position():
    for (answer, text) in zip(val_answers,val_texts):
        real_answer = answer['text']
        start_index = answer['answer_start']
         # Get the real end index
        end_index = start_index + len(real_answer)

        # S’assurer que la chaîne real_answer correspond bien à un extrait du text (le contexte), en ajustant l’index si besoin.
        if text[start_index:end_index] == real_answer:
            answer['answer_end'] = end_index
        elif text[start_index-1:end_index-1] == real_answer:
            answer['answer_start'] = start_index - 1
            answer['answer_end'] = end_index - 1
        elif text[start_index-2:end_index-2] == real_answer:
            answer['answer_start'] = start_index - 2
            answer['answer_end'] = end_index - 2

find_start_end_validation_answer_position()
# Tokenize passages and queries
train_encodings  = tokenizer(train_texts,train_queries, truncation = True, padding= True)
val_encodings = tokenizer(val_texts, val_queries, truncation = True, padding = True)

#{
#  'input_ids': [[101, 2054, 2003, 1996, 3007, 1997, 2605, 1029, 102, 3000, 2003, 1996, 3007, 1997, 2605, 1012, 102]],
#  'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]],
#  'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
#}

#input_ids : IDs des tokens du texte + question, concaténés. Ces IDs sont les entrées principales du modèle BERT.
#token_type_ids : Indique quelles parties sont la question (0) et quelles parties sont le contexte (1).
#attention_mask : Valeur 1 pour les tokens à garder, 0 pour ceux à ignorer (padding).


# Convert the start-end positions to tokens start-end positions
def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    
    count = 0
    for i in (len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))
        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        # if end position is None, the 'char_to_token' function points to the space after the correct token, so add - 1
        if end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - 1)
            
            # if end position is still None the answer passage has been truncated
            if end_positions[-1] is None:
                count += 1
                end_positions[-1] = tokenizer.model_max_length

    print(count)

    # Update the data in dictionary
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)

# Avant de passer les données dans BERT, tu dois :
# Tokeniser le contexte et la question.
# Encoder les positions de la réponse (start/end).
# Convertir tout ça en tensors PyTorch.

# Create a Dataset class
train_dataset = SquadDataset(train_encodings.select(range(6)))
val_dataset = SquadDataset(val_encodings.select(range(6)))

# Use of DataLoader
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=True)

# Train and Evaluate Model
optim = AdamW(model.parameters(), lr=5e-5)
epochs = 3
whole_train_eval_time = time.time()
train_losses = []
val_losses = []

print_every = 6

for epoch in range(epochs):
    epoch_time = time.time()
    
    # Set model in train mode
    model.train()
    total_train_loss = 0
    
    print("############Train############")
    
    for batch_idx,batch in enumerate(train_loader): 
        optim.zero_grad()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        
        loss, _, _  = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        # do a backwards pass 
        loss.backward()
        # update the weights
        optim.step()
        # Find the total loss
        total_train_loss += loss.item()
        
        if (batch_idx+1) % print_every == 0:
            print("Batch {:} / {:}".format(batch_idx+1,len(train_loader)),"\nLoss:", round(loss.item(),1),"\n")
        
    total_train_loss /= len(train_loader)
    train_losses.append(total_train_loss)
    
    ##########Evaluation##################
    
    # Set model in evaluation mode
    model.eval()
    
    print("############Evaluate############")
    
    total_val_loss = 0
    
    for batch_idx,batch in enumerate(val_loader):
        
        with torch.no_grad():
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            start_positions = batch['start_positions']
            end_positions = batch['end_positions']
            
            loss, _, _ = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            total_val_loss += loss.item()
            
        if (batch_idx+1) % print_every == 0:
            print("Batch {:} / {:}".format(batch_idx+1,len(val_loader)),"\nLoss:", round(loss.item(),1),"\n")
    
    total_val_loss /= len(val_loader)
    val_losses.append(total_val_loss)
    
    # Print each epoch's time and train/val loss 
    
    print("\n-------Epoch ", epoch+1,
          "-------"
          "\nTraining Loss Avg:", train_losses[-1],
          "\nValidation Loss Avg:", val_losses[-1],
          "\nTime: ",(time.time() - epoch_time),
          "\n-----------------------",
          "\n\n")

print("Total training and evaluation time: ", (time.time() - whole_train_eval_time))

# Cette fonction fait une prédiction de réponse à partir d’un contexte (texte) et d’une question.
def predict(context,query):
    
    # query (la question) est mise avant context (le passage) comme c’est la norme en QA.
    inputs = tokenizer.encode_plus(query, context, return_tensors='pt')
    outputs = model(**inputs)
    # On prend l’indice avec la valeur la plus élevée = prédiction la plus probable.
    # get the most likely beginning of answer with the argmax of the score
    answer_start = torch.argmax(outputs[0]) # index du début de la réponse 
    answer_end = torch.argmax(outputs[1]) + 1 # index de la fin de la réponse
    # Convertit les tokens (IDs) en mots pour obtenir une réponse lisible
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    return answer

# Fonction utilitaire pour normaliser un texte :
# minuscule
# suppression des articles ("a", "an", "the")
# suppression de la ponctuation
# espaces propres
# Utilisée pour comparer proprement deux textes, comme les réponses
def normalize_text(s):
    # Removing articles and punctuation, and standardizing whitespace are all typical text processing steps
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# Retourne 1 si la prédiction correspond exactement à la vérité, 0 sinon, après normalisation
def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

# Utilisé pour évaluer la qualité partielle d’une réponse, même si elle n’est pas parfaite
def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    # Si une des deux chaînes est vide, on ne peut pas calculer un score F1 car diviser par 0 ou comparer des mots n’a pas de sens
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens) # 1 si les deux sont vides (ils sont égaux) sinon 0
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    # basé sur les mots en commun entre prédiction et vérité
    # F1 = 2 * (précision * rappel) / (précision + rappel)
    return 2 * (prec * rec) / (prec + rec)

def give_an_answer(context,query,answer):
    # prédire une réponse
    prediction = predict(context,query)
    em_score = compute_exact_match(prediction, answer)
    f1_score = compute_f1(prediction, answer)
    # afficher la question, prédiction, réponse vraie
    # afficher les scores EM (Exact Match) et F1
    print(f"Question: {query}")
    print(f"Prediction: {prediction}")
    print(f"True Answer: {answer}")
    print(f"EM: {em_score}")
    print(f"F1: {f1_score}")
    print("\n")

# Build the Bert model
if __name__ == '__main__':
    context = "Hi! My name is Alexa and I am 21 years old. I used to live in Peristeri of Athens, but now I moved on in Kaisariani of Athens."

    queries = ["How old is Alexa?",
            "Where does Alexa live now?",
            "Where Alexa used to live?"
            ]
    answers = ["21",
            "Kaisariani of Athens",
            "Peristeri of Athens"
            ]

    for q,a in zip(queries,answers):
        give_an_answer(context,q,a)