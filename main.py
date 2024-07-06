import torchtext
torchtext.disable_torchtext_deprecation_warning()

import numpy as np
import pandas as pd
import re
import warnings
import torch
import os
import torch.nn as nn
from tqdm import tqdm
from torchtext.vocab import GloVe
from collections import Counter
from torchtext.vocab import GloVe
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
import warnings

warnings.filterwarnings("ignore")

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attn_weights = F.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context, attn_weights

class SentimentClassifierLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, output_dim, dropout):
        super(SentimentClassifierLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_output, (hidden, _) = self.lstm(embedded)
        context, attn_weights = self.attention(lstm_output)
        hidden = self.dropout(context)
        hidden = torch.relu(self.fc1(hidden))
        output = self.fc2(hidden)
        return output

def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_text(text):
    tokens = text.split()
    return [glove.stoi[token] for token in tokens if token in glove.stoi]


def predict_sentiment_LSTM(text):
    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    text = clean_text(text)
    tokens = tokenize_text(text)
    padded_tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
    input_tensor = torch.tensor(padded_tokens, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        output = model_LSTM(input_tensor)
        _, predicted = torch.max(output, 1)
    return sentiment_mapping[predicted.item()]


def predict_sentiment_XLMRoBERTa(text):
    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

    inputs = tokenizer_xlm_roberta(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model_xlm_roberta(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(predictions, dim=1).item()
    sentiment_label = sentiment_mapping[sentiment]

    return sentiment_label


def predict_sentiment_BERT(text):
    sentiment_lower = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}

    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

    inputs = tokenizer_BERT(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model_BERT(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(predictions, dim=1).item()
    sentiment_label = sentiment_mapping[sentiment_lower[sentiment]]

    return sentiment_label


def get_majority_label(predictions):
    counter = Counter(predictions)
    majority_label = counter.most_common(1)[0][0]
    return majority_label


def predict_sentiment_ensemble(text):
    prediction_LSTM = predict_sentiment_LSTM(text)
    prediction_XLMRoBERTa1 = predict_sentiment_XLMRoBERTa(text)
    prediction_XLMRoBERTa2 = predict_sentiment_XLMRoBERTa(text)
    prediction_BERT = predict_sentiment_BERT(text)

    return get_majority_label((prediction_LSTM, prediction_XLMRoBERTa1, prediction_XLMRoBERTa2, prediction_BERT))

base_dir = "D:\\BSES - Data Analyst\\Sentiment Analysis"

processed_filepath = os.path.join(base_dir, "newData", "New Processed Data.xlsx")
universal_filepath = os.path.join(base_dir, "newData", "Universal.xlsx")
custom_embed_dir = os.path.join(base_dir, "Embedding")
LSTM_path = os.path.join(base_dir, "Models", "LSTM.pt")
custom_cache_dir = os.path.join(base_dir, "Models")
to_process_path = os.path.join(base_dir, "newData", "To Process")

# Model 1: LSTM + Attention
print("Loading Model 1: LSTM + Attention Model")
max_len = 500
embed_dim = 300
batch_size = 64
hidden_dim = 100
output_dim = 3
dropout_rate = 0.2
num_layers = 2
glove = GloVe(name='6B', dim=embed_dim, cache=custom_embed_dir)
vocab_size = len(glove.stoi)

model_LSTM = SentimentClassifierLSTM(vocab_size, embed_dim, hidden_dim, num_layers, output_dim, dropout_rate)
model_LSTM.load_state_dict(torch.load(LSTM_path))
model_LSTM.eval()
print("Model 1 Loaded\n")

# XLM-RoBERTa Model
print("Loading Model 2: XLM-RoBERTa Model")
model_name_RoBERTa = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
os.makedirs(custom_cache_dir, exist_ok=True)
tokenizer_xlm_roberta = AutoTokenizer.from_pretrained(model_name_RoBERTa, cache_dir=custom_cache_dir)
model_xlm_roberta = AutoModelForSequenceClassification.from_pretrained(model_name_RoBERTa, cache_dir=custom_cache_dir)
print("Model 2 Loaded\n")

# BERT Model
print("Loading Model 3: BERT Model")
model_name_BERT = 'nlptown/bert-base-multilingual-uncased-sentiment'
os.makedirs(custom_cache_dir, exist_ok=True)
tokenizer_BERT = BertTokenizer.from_pretrained(model_name_BERT, cache_dir=custom_cache_dir)
model_BERT = BertForSequenceClassification.from_pretrained(model_name_BERT, cache_dir=custom_cache_dir)
print("Model 3 Loaded\n")

universal = pd.read_excel(universal_filepath, engine='openpyxl')
last_date_universal = universal['Date'].max()
print(f"Last Data in Universal: {last_date_universal}\n")

to_process_file_name = os.listdir(to_process_path)
to_process_file_path = os.path.join(to_process_path, to_process_file_name[0])
print(f"Reading Excel file from path: {to_process_path}")
print(f"Filename : {to_process_file_name[0]}")
df = pd.read_excel(to_process_file_path, engine='openpyxl')
if last_date_universal is np.nan:
    data = df["Customer_Text"].copy()
else:
    filtered_data = df[df['Date'] > last_date_universal].reset_index(drop=True)
    data = filtered_data["Customer_Text"].copy()
data = data.astype(str)


sentiment = []
for text in tqdm(data, desc="Processing Sentiments"):
    text = text.replace('\n', ' ').strip()
    sentiment.append(predict_sentiment_ensemble(text))

filtered_data["Sentiment"] = sentiment

universal = pd.read_excel(universal_filepath, engine='openpyxl')
new_universal = pd.concat([universal, filtered_data], ignore_index=True)
new_universal.to_excel(universal_filepath, index=False)
filtered_data.to_excel(processed_filepath, index=False)

input("\nPress any key to exit!")