###############################################################################
# A4-RNN-your_name_connected_by_underscore-your_student_id.py
#
# Usage example (from command line):
#   python A4-RNN-your_name_connected_by_underscore-your_student_id.py /path/to/test_dataset.csv
#
# Make sure you have saved your trained model in the same directory as:
#   A4-RNN-your_name_connected_by_underscore-your_student_id.pth
#
# Then this script will:
#   1) Load the RNN model definition.
#   2) Load model weights from the .pth file.
#   3) Load and preprocess the dataset from the CSV file path provided.
#   4) Predict sentiment (0 or 1) for each sample.
#   5) Print the predictions surrounded by special sentinel lines for autograding.
###############################################################################

import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from collections import Counter
import re

def preprocess_string(str1):
    # remove all non-word characters excluding number and letters
    str1= re.sub(r"[^\w\s]",'',str1)
    # remove all whitespace with no space
    str1= re.sub(r"\s",'',str1)
    # replace digits with no space
    str1= re.sub(r"\d",'',str1)
    return str1

def preprocess_sentence(sen1):
    word_list=[]
    stop_word = set(stopwords.words("english"))
    for word in sen1.lower().split():
        word = preprocess_string(word)
        if word not in stop_word and word!='':
            word_list.append(word)
    return word_list

def get_stoi(data):
    word_list=[]
    for review in data:
        word_list.extend(preprocess_sentence(review))
    corpus = Counter(word_list)
    print(corpus.get)
    # sorting on the basis of most common words
    corpus_ =sorted(corpus,key= corpus.get,reverse=True)[:1000]
    # creating a dict
    stoi =  {ch:i+1 for i,ch in enumerate(corpus_)}
    return stoi

def tokenize(data, labels, stoi):
    # tokenize
    data_encoded = []
    for review in data:
        data_encoded.append([stoi[word] for word in preprocess_sentence(review)
                             if word in stoi.keys()])

    labels_encoded = [1 if label =='positive' else 0 for label in labels]

    return np.array(data_encoded, dtype=object), np.array(labels_encoded)

df = pd.read_csv("IMDB Dataset.csv")

# TODO: Adjust this to match what you have in your A4.ipynb Part A - 1 Data Cleaning
df_train, _ = train_test_split(df, test_size=0.4, random_state=42)
stoi = get_stoi(df_train['review'].values)

###############################################################################
# 1. Define/Implement the SentimentRNN Model
###############################################################################
# TODO: copy your RNN model definition from A4.ipynb Part A - 2 Model Building

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=2, output_dim=1, dropout=0.5, pooling='concat'):
        super(SentimentRNN, self).__init__()

        # TO BE COMPLETED
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.pooling = pooling
        if pooling == 'concat':
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden=None):
        if x.size(1) == 0:
            print("Error: Received an empty input sequence")
        # TO BE COMPLETED
        embedded = self.embedding(x)
        out, hidden = self.lstm(embedded, hidden)
        
        if self.pooling == 'last':
            out = out[:, -1, :]
        elif self.pooling == 'max':
            out, _ = torch.max(out, dim=1) 
        elif self.pooling == 'concat':
            max_pool, _ = torch.max(out, dim=1)
            avg_pool = torch.mean(out, dim=1)
            out = torch.cat([max_pool, avg_pool], dim=1)
        
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out)

def prepare_model():
    # TODO: declare your model here, e.g. SentimentRNN(your parameters)
    # Create model instance with the SAME hyperparameters used in training
    # so that loading the state_dict will work correctly.
    # For example:
    # model = SentimentRNN(vocab_size=10000, embed_dim=300, hidden_dim=128, output_dim=1, n_layers=2)
    # Students can change the parameters based on their model
    # Hyperparameters for RNN
    vocab_size = len(stoi) + 1
    embedding_dim = 200
    hidden_dim = 64
    num_layers = 4
    output_dim = 1
    dropout = 0.7
    pooling = 'concat'
    model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, num_layers, output_dim, dropout, pooling)
    
    return model

###############################################################################
# 2. Helper function to preprocess or tokenize text
###############################################################################
# TODO: Add all the preprocessing code you need for the model to work

def preprocess_text(review, padding_length=500):
    encoded = [stoi[word] for word in preprocess_sentence(review) if word in stoi.keys()]
    if len(encoded) < padding_length:
        padded = [0] * (padding_length - len(encoded)) + encoded
    else:
        padded = encoded[:padding_length]

    return padded
###############################################################################
# 3. Helper function to load and prepare your dataset for inference
###############################################################################
def load_dataset(csv_path):
    """
    Reads the CSV file from csv_path and returns a list of texts or
    a DataFrame with a column that the student can map to indices for the RNN.
    """
    # Example: using pandas
    df = pd.read_csv(csv_path)
    return df

###############################################################################
# 4. Main: parse arguments, load model & data, generate predictions
###############################################################################
def main():
    # -------------------------------------------------------------------------
    # A) Parse the command-line argument for the dataset path
    # -------------------------------------------------------------------------
    if len(sys.argv) < 2:
        print("Usage: python A4-RNN-your_name_connected_by_underscore-your_student_id.py /path/to/test_dataset.csv")
        sys.exit(1)
    dataset_path = sys.argv[1]
    
    # -------------------------------------------------------------------------
    # B) Initialize your model and load the saved model weights
    # -------------------------------------------------------------------------
    # Make sure the .pth file name matches the script name, e.g.:
    # Get the script name and replace .py with .pth for the weights file
    model_weights_path = __file__.replace('.py', '.pth') 
    
    model = prepare_model()
    # Load the model weights
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cuda')))
    model.eval()
    
    # -------------------------------------------------------------------------
    # C) Load the dataset, preprocess, and prepare for inference
    # -------------------------------------------------------------------------
    df = load_dataset(dataset_path)

    # Example assumption: df has a 'review_text' column
    # We'll store predictions in a list for each row
    all_predictions = []
    
    for i in range(len(df)):
        # 1) Read text
        text = df.iloc[i]['review']  # adapt to the correct column name
        # 2) Preprocess to indices
        text_indices = preprocess_text(text)
        # 3) Convert to torch tensor
        text_tensor = torch.tensor(text_indices).unsqueeze(0)  # shape: (1, seq_len)
        
        # 4) Forward pass to get the output
        with torch.no_grad():
            outputs = model(text_tensor)
            predicted = (outputs > 0.5).float()  # get the predicted class
            predicted_label = int(predicted.item())  # convert to scalar
        
        # ensure only 0 or 1
        assert predicted_label in [0, 1], "Prediction must be 0 or 1"
        
        all_predictions.append(predicted_label)
    
    # -------------------------------------------------------------------------
    # D) Print predictions between sentinel lines for autograder
    # -------------------------------------------------------------------------
    print("===start_output===")
    for pred in all_predictions:
        print(pred)
    print("===end_output===")

###############################################################################
# Entry point
###############################################################################
if __name__ == "__main__":
    main()
