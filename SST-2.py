"""
For Hydrogen;
%load_ext autoreload
%autoreload 2

Probe BERT hidden representations for SST (Linguistic correlation analysis)
"""

from pathlib import Path
import pandas as pd
import torch
from pytorch_transformers import BertForSequenceClassification, BertTokenizer


model_path = Path("models/SST-2/")
model = BertForSequenceClassification.from_pretrained(str(model_path))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Error analysis
dev_data_path = Path("glue_data/SST-2/dev.tsv")
df = pd.read_csv(dev_data_path,delimiter="\t")
sentences = df["sentence"].values.tolist()

## Preprocess to bert format
pred_list = []
most_long = 0
for s in sentences:  # s = sentences[0]
    s = f"[CLS] {s} [SEP]"
    encoded = torch.LongTensor(tokenizer.encode(s))
    encoded = encoded.reshape(1,len(encoded))
    pred = model(encoded)[0].tolist()[0]
    pred_list.append(pred)

## Extract sentence where the model failed
df["pred"] = list(map(lambda x: 0 if x[0]>x[1] else 1,pred_list))
df[df["pred"] != df["label"]].to_csv("results/SST-2/error_analysis.csv")
