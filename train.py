# %% 
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "bhadresh-savani/bert-base-uncased-emotion"
model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
