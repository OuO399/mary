from flask import Flask,request
from flask import jsonify
import sys
import os
import json
import collections
import string
import javalang
import torch
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline,RobertaModel,AutoTokenizer,AutoModelForMaskedLM

os.environ['TORCH_USE_CUDA_DSA'] = '1'

GRAPH_CODEBERT_MLM="pre-trained/graphcodebert-base"
CODEBERT_MLM="pre-trained/codebert-base-mlm"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_graphc1 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
model_graphc1 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
fill_mask_graphc1 = pipeline('fill-mask', model=model_graphc1, tokenizer=tokenizer_graphc1, device=0 if device.type == "cuda" else -1)

tokenizer_graphc2 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
model_graphc2 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
fill_mask_graphc2 = pipeline('fill-mask', model=model_graphc2, tokenizer=tokenizer_graphc2, device=0 if device.type == "cuda" else -1)

tokenizer_graphc3 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
model_graphc3 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
fill_mask_graphc3 = pipeline('fill-mask', model=model_graphc3, tokenizer=tokenizer_graphc3, device=0 if device.type == "cuda" else -1)

tokenizer_graphc4 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
model_graphc4 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
fill_mask_graphc4 = pipeline('fill-mask', model=model_graphc4, tokenizer=tokenizer_graphc4, device=0 if device.type == "cuda" else -1)

tokenizer_graphc5 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
model_graphc5 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
fill_mask_graphc5 = pipeline('fill-mask', model=model_graphc5, tokenizer=tokenizer_graphc5, device=0 if device.type == "cuda" else -1)

tokenizer_graphc6 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
model_graphc6 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
fill_mask_graphc6 = pipeline('fill-mask', model=model_graphc6, tokenizer=tokenizer_graphc6, device=0 if device.type == "cuda" else -1)

tokenizer_graphc7 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
model_graphc7 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
fill_mask_graphc7 = pipeline('fill-mask', model=model_graphc7, tokenizer=tokenizer_graphc7, device=0 if device.type == "cuda" else -1)

tokenizer_graphc8 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
model_graphc8 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
fill_mask_graphc8 = pipeline('fill-mask', model=model_graphc8, tokenizer=tokenizer_graphc8, device=0 if device.type == "cuda" else -1)

tokenizer_graphc9 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
model_graphc9 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
fill_mask_graphc9 = pipeline('fill-mask', model=model_graphc9, tokenizer=tokenizer_graphc9, device=0 if device.type == "cuda" else -1)

tokenizer_graphc10 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
model_graphc10 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
fill_mask_graphc10 = pipeline('fill-mask', model=model_graphc10, tokenizer=tokenizer_graphc10, device=0 if device.type == "cuda" else -1)

# tokenizer_graphc11 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
# model_graphc11 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
# fill_mask_graphc11 = pipeline('fill-mask', model=model_graphc11, tokenizer=tokenizer_graphc11, device=0 if device.type == "cuda" else -1)

# tokenizer_graphc12 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
# model_graphc12 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
# fill_mask_graphc12 = pipeline('fill-mask', model=model_graphc12, tokenizer=tokenizer_graphc12, device=0 if device.type == "cuda" else -1)

# tokenizer_graphc13 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
# model_graphc13 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
# fill_mask_graphc13 = pipeline('fill-mask', model=model_graphc13, tokenizer=tokenizer_graphc13, device=0 if device.type == "cuda" else -1)

# tokenizer_graphc14 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
# model_graphc14 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
# fill_mask_graphc14 = pipeline('fill-mask', model=model_graphc14, tokenizer=tokenizer_graphc14, device=0 if device.type == "cuda" else -1)

# tokenizer_graphc15 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
# model_graphc15 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
# fill_mask_graphc15 = pipeline('fill-mask', model=model_graphc15, tokenizer=tokenizer_graphc15, device=0 if device.type == "cuda" else -1)

# tokenizer_graphc16 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
# model_graphc16 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
# fill_mask_graphc16 = pipeline('fill-mask', model=model_graphc16, tokenizer=tokenizer_graphc16, device=1 if device.type == "cuda" else -1)

# tokenizer_graphc17 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
# model_graphc17 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
# fill_mask_graphc17 = pipeline('fill-mask', model=model_graphc17, tokenizer=tokenizer_graphc17, device=1 if device.type == "cuda" else -1)

# tokenizer_graphc18 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
# model_graphc18 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
# fill_mask_graphc18 = pipeline('fill-mask', model=model_graphc18, tokenizer=tokenizer_graphc18, device=1 if device.type == "cuda" else -1)

# tokenizer_graphc19 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
# model_graphc19 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
# fill_mask_graphc19 = pipeline('fill-mask', model=model_graphc19, tokenizer=tokenizer_graphc19, device=1 if device.type == "cuda" else -1)

# tokenizer_graphc20 = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
# model_graphc20 = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
# fill_mask_graphc20 = pipeline('fill-mask', model=model_graphc20, tokenizer=tokenizer_graphc20, device=1 if device.type == "cuda" else -1)


# tokenizer_codebert = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
# model_codebert = RobertaForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM).to(device)
# fill_mask_codebert = pipeline('fill-mask', model=model_codebert, tokenizer=tokenizer_codebert, device=0 if device.type == "cuda" else -1)
# tokenizer_graphc_cpu = RobertaTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
# model_graphc_cpu = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM)
# fill_mask_graphc_cpu = pipeline('fill-mask', model=model_graphc_cpu, tokenizer=tokenizer_graphc_cpu)