# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer,BertModel,AutoTokenizer,AutoModelForMaskedLM
from torch.utils import data

from collections import Counter

import ast

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

def calculate_dipeptide_composition(sequence):
    sequence = ''.join([aa for aa in sequence if aa in amino_acids])
    dipeptides = [sequence[i:i+2] for i in range(len(sequence)-1)]
    dipeptide_counts = Counter(dipeptides)
    num_dipeptides = len(amino_acids) ** 2
    dipeptide_matrix = np.zeros(num_dipeptides)
    
    for dipeptide, count in dipeptide_counts.items():
        index = amino_acids.index(dipeptide[0]) * len(amino_acids) + amino_acids.index(dipeptide[1])
        dipeptide_matrix[index] = count
    return dipeptide_matrix

p_tokenizer = BertTokenizer.from_pretrained("./biobert-v1.2")
p_model = BertModel.from_pretrained("./biobert-v1.2")
d_tokenizer = AutoTokenizer.from_pretrained("./chamberts")
d_model = AutoModelForMaskedLM.from_pretrained("./chamberts")

for param in p_model.parameters():
    param.requires_grad = False

for param in d_model.parameters():
    param.requires_grad = False

max_d = 205
max_p = 512
linear_layer = nn.Linear(600, 768)

df = pd.read_csv("./dataset/sBioSNAP/all.csv")

def protein2emb_encoder(x):

    output=""
    for char in x:
        output += char + " "
    outputs = output.strip()
    inputs = p_tokenizer.encode_plus(outputs, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    with torch.no_grad():
        outputs = p_model(**inputs)

    protein_features = outputs.last_hidden_state
    protein_features = np.squeeze(protein_features)

    array = torch.ones(513)

    dc_features = calculate_dipeptide_composition(x)
    print("Dipeptide Composition Features:", dc_features)
    print("Feature Vector Shape:", dc_features.shape)
    input_tensor = torch.tensor(dc_features)
    padded_tensor = torch.nn.functional.pad(input_tensor, (0, 368)).unsqueeze(0)
    protein_features = torch.cat((protein_features, padded_tensor), dim=0)

    return protein_features, array

def drug2emb_encoder(x,y):
    inputs = d_tokenizer(x, return_tensors="pt", max_length=70, padding="max_length", truncation=True)

    with torch.no_grad():
        outputs = d_model(**inputs)

    protein_features = outputs.logits
    new_x1 = linear_layer(protein_features.unsqueeze(2)).squeeze(2)

    query_result = df[(df['SMILES'] == x) & (df['Target Sequence'] == y)]
    if query_result.empty:
        raise ValueError(f"notfound SMILES={x} & Target Sequence={y} in CSV")

    ecfp_str = query_result.iloc[0]['ECFP'] 
    ecfp_list = ast.literal_eval(ecfp_str) 
    ecfp_tensor = torch.tensor(ecfp_list, dtype=torch.float32).unsqueeze(0) 

    device = new_x1.device  
    ecfp_tensor = ecfp_tensor.to(device) 
    print(new_x1.shape)
    print(ecfp_tensor.shape)
    new_x1=new_x1.squeeze(0)
    final_tensor = torch.cat((new_x1, ecfp_tensor), dim=0)  
    array = torch.ones(71)
    return final_tensor, array

class BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        d = self.df.iloc[index]['SMILES']
        p = self.df.iloc[index]['Target Sequence']
        p_v, input_mask_p = protein2emb_encoder(p)
        d_v, input_mask_d = drug2emb_encoder(d,p)
        print("in stream")
        y = self.labels[index]
        y = torch.tensor(y, dtype=torch.float32)
        return d_v, p_v, input_mask_d, input_mask_p, y
