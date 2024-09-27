import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from transformers import BertModel
import torch.nn as nn
import joblib
import os
import re
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is set to:", device)

class MultiTaskBERT(LightningModule):
    def __init__(self, num_classes_list, class_weights, dropout_rate, hidden_size):
        super(MultiTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_layer = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.classification_heads = nn.ModuleList([nn.Linear(hidden_size, num_classes) for num_classes in num_classes_list])
        self.class_weights = class_weights
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        hidden_output = self.activation(self.hidden_layer(cls_output))
        logits = [head(hidden_output) for head in self.classification_heads]
        return logits

class IndicatorDataset(Dataset):
    def __init__(self, dataframe, tokenizer, target_columns):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.target_columns = target_columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        tokenized = item['tokenized']
        input_ids = torch.tensor(tokenized['input_ids']).squeeze()
        attention_mask = torch.tensor(tokenized['attention_mask']).squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

def load_model_and_predict(model_path, input_file, output_file, confidence_threshold=0.8):
    # Load num_classes_list
    num_classes_list = joblib.load('num_classes_list.pkl')
    class_weights = [None] * len(num_classes_list)  # Dummy weights, not used for inference
    
    # Load the model
    model = MultiTaskBERT.load_from_checkpoint(model_path, num_classes_list=num_classes_list, class_weights=class_weights, dropout_rate=0.20615965554535598, hidden_size=512)
    model.eval()
    model.to(device)
    
    # Load and preprocess the dataset
    dataset = pd.read_excel(input_file)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dataset['combined_features'] = dataset.apply(lambda row: ' '.join([
    row['feature_1'],
    row['feature_2'] if pd.notna(row['feature_2']) else '',
    row['feature_3'] if pd.notna(row['feature_3']) else '',
    row['feature_4'] if pd.notna(row['feature_4']) else ''
    ]), axis=1)

    dataset['tokenized'] = dataset['combined_features'].apply(lambda x: tokenizer.encode_plus(
        x, add_special_tokens=True, max_length=128, padding='max_length', truncation=True, return_tensors='pt'
    ))

    dataset['tokenized'] = dataset['tokenized'].apply(lambda x: {
        'input_ids': x['input_ids'].tolist(),
        'attention_mask': x['attention_mask'].tolist()
    })

    target_columns = [
    'target_1', 'target_2', 'target_3', 
    'target_4', 'target_5', 'target_6', 'target_7', 
    'target_8', 'target_9', 'target_10', 'target_11', 
    'target_12', 'target_13', 'target_14', 'target_15'
    ]

    # Load label encoders
    label_encoder_dir = 'label_encoders'
    label_encoders = {col: joblib.load(os.path.join(label_encoder_dir, f'label_encoder_{re.sub(r"[^a-zA-Z0-9_]", "_", col)}.pkl')) for col in target_columns}

    test_dataset = IndicatorDataset(dataset, tokenizer, target_columns)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    all_preds = []
    all_confidences = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask)
            batch_preds = []
            batch_confidences = []
            for logit in logits:
                probs = F.softmax(logit, dim=1)
                preds = torch.argmax(probs, dim=1)
                confidences = probs[range(len(preds)), preds]

                batch_preds.append(preds.cpu().numpy())
                batch_confidences.append(confidences.cpu().numpy())
            
            all_preds.append(np.stack(batch_preds, axis=1))
            all_confidences.append(np.stack(batch_confidences, axis=1))

    # Concatenate results properly
    all_preds = np.concatenate(all_preds, axis=0)
    all_confidences = np.concatenate(all_confidences, axis=0)
    
    flag_column = []
    for i, col in enumerate(target_columns):
        dataset[col] = all_preds[:, i]
        dataset[f'{col}_confidence'] = all_confidences[:, i]
        below_threshold = all_confidences[:, i] < confidence_threshold
        flag_column.append(below_threshold)
        
        # Replace numerical labels with original labels
        try:
            dataset[col] = label_encoders[col].inverse_transform(dataset[col])
        except ValueError as e:
            print(f"Error in column {col}: {e}")
            unique_values = dataset[col].unique()
            print(f"Unique values in predictions for {col}: {unique_values}")
            print(f"Classes in encoder for {col}: {label_encoders[col].classes_}")
            return

    # Create the flag column
    dataset['flag'] = np.any(np.column_stack(flag_column), axis=1).astype(int)

    # Drop unwanted columns
    dataset = dataset.drop(columns=['combined_features', 'tokenized'], errors='ignore')

    # Save the results to a new Excel file
    dataset.to_excel(output_file, index=False)

# Usage example
model_path = 'uc3_model.ckpt'
input_file = 'test.xlsx'
output_file = 'output.xlsx'

load_model_and_predict(model_path, input_file, output_file)

print("Prediction completed")
