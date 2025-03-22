###############################################################################
# A4-BERT-your_name_connected_by_underscore-your_student_id.py
#
# Usage example (from command line):
#   python A4-BERT-your_name_connected_by_underscore-your_student_id.py /path/to/test_dataset.csv
#
# This script will:
#   1) Load the BERT-based sentiment classifier definition.
#   2) Load model weights from the .pth file (with the same stem as this .py file).
#   3) Load and preprocess the dataset from the CSV file path provided.
#   4) Predict sentiment (0 or 1) for each sample.
#   5) Print the predictions surrounded by special sentinel lines for autograding.
###############################################################################

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import BertModel, BertTokenizer

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
MAX_LEN = 400

###############################################################################
# 1. Define your BERT-based model
###############################################################################
# TODO: copy your Best BERT-Based model definition from A4.ipynb
# Your best model is either SentimentClassifierPooled or SentimentClassifierLast
# Just copy the model definition here
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        # More complex architecture
        self.fc1 = nn.Linear(768, 512)
        self.bn1 = nn.BatchNorm1d(512) 
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.3) 
        
    def forward(self, pooled_embedding):
        x = self.fc1(pooled_embedding)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x.squeeze(1)

def prepare_model():
    # TODO: declare your model here, e.g. SentimentClassifier(your parameters)
    # Create model instance with the SAME hyperparameters used in training
    # so that loading the state_dict will work correctly.
    model = SentimentClassifier(n_classes=1)
    return model


# TODO: Change this based on the input your model expects
EMBEDDINGS_TYPE = 'pooled'  # or 'last_hidden_state'

###############################################################################
# 2. Helper function to load/preprocess your dataset
###############################################################################
def load_dataset(csv_path):
    """
    Reads the CSV file from csv_path and returns a DataFrame. 
    Assumes there's a column named 'review' or similar.
    """
    df = pd.read_csv(csv_path)
    return df

def preprocess_text(tokenizer, review):
    """
    Tokenize and encode a single text string with the specified tokenizer.
    Returns the (input_ids, attention_mask) as torch tensors with shape (1, max_length).
    """
    encoded = tokenizer.encode_plus(
        str(review),
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    embeddings = bert_model(**encoded, output_hidden_states=True)
    if EMBEDDINGS_TYPE == 'pooled':
        return embeddings.pooler_output.detach().cpu()
    elif EMBEDDINGS_TYPE == 'last_hidden_state':
        return embeddings.hidden_states[-1].detach().cpu()
    else:
        raise ValueError("EMBEDDINGS_TYPE must be 'pooled' or 'last_hidden_state'")

###############################################################################
# 3. Main: parse arguments, load model & data, generate predictions
###############################################################################
def main():
    # -------------------------------------------------------------------------
    # A. Parse the command-line argument for the dataset path
    # -------------------------------------------------------------------------
    if len(sys.argv) < 2:
        print("Usage: python A4-BERT-your_name_connected_by_underscore-your_student_id.py /path/to/test_dataset.csv")
        sys.exit(1)
    dataset_path = sys.argv[1]

    # -------------------------------------------------------------------------
    # B. Initialize your model and load the saved model weights
    # -------------------------------------------------------------------------

    # Create an instance of your model with the SAME hyperparameters used in training
    model = prepare_model()

    # Derive the model weights file name from this script name
    model_weights_path = __file__.replace('.py', '.pth')

    # Load the model weights
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()

    # -------------------------------------------------------------------------
    # C. Load the dataset, preprocess, and prepare for inference
    # -------------------------------------------------------------------------
    df = load_dataset(dataset_path)

    all_predictions = []

    for i in range(len(df)):
        # 1. Read text
        review = df.iloc[i]['review']  # adapt the column name if it's not 'review'
        
        # 2. Preprocess -> get embeddings
        embeddings = preprocess_text(tokenizer, review)
        
        # 3. Forward pass
        with torch.no_grad():
            outputs = model(embeddings)
            prob = torch.sigmoid(outputs).item()
            predicted_label = int(prob > 0.5)

        # Make sure the label is 0 or 1
        assert predicted_label in [0, 1], "Prediction must be 0 or 1"
        all_predictions.append(predicted_label)

    # -------------------------------------------------------------------------
    # D. Print predictions between sentinel lines for autograder
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
