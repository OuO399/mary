#!/usr/bin/python

import sys
import os

import json
import collections
import string
import javalang
import torch
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline,RobertaModel,AutoTokenizer,AutoModelForMaskedLM
import requests
import multiprocessing

os.environ['TORCH_USE_CUDA_DSA'] = '1'

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    f = open(vocab_file,)
    reader = json.load(f) 
    for token in reader.keys():
        index = reader[token]
        token = token.encode("ascii", "ignore").decode()
        token = ''.join(token.split())
        #print("token: '",token,"'")
        vocab[index] = token
    f.close()
    return vocab


CODE = sys.argv[1]
method_code = sys.argv[2]
PROCESS_LABEL = sys.argv[3]
# print("----------------------------",multiprocessing.current_process().name.split()[1])
# Load vocab file
my_dict = load_vocab('pre-trained/graphcodebert-base/vocab.json')

SHRINKED_CODE_JSON = "{'masked_seq':" + repr(CODE) + "}"
# SHRINKED_CODE_JSON = "{'ast_seq':" + repr(method_ast_token) + ",'masked_seq':"+ repr(CODE) +"}"
# SHRINKED_CODE_JSON = "{'masked_seq':" + repr(tmp) + "}"
print(SHRINKED_CODE_JSON)
url = f"http://127.0.0.1:8088/graphc_use_ast{PROCESS_LABEL}"
input_data ={"data":{"CODE":CODE,"method_code":method_code,"debug":multiprocessing.current_process().name}}
res = json.loads(requests.post(url,data=json.dumps(input_data)).text)
outputs = res["data"]

for out in outputs:
	json_str = json.dumps(out)
	json_object = json.loads(json_str)
	token_str_exists = False; #"token_str" in json_object
	if not token_str_exists:
		index = json_object["token"]
		token_str = my_dict[index] 
		token_str = token_str.encode("ascii", "ignore").decode()
		json_object['token_str'] = token_str
	print(json_object)

  

  








# import sys
# import json
# import collections
# import string

# # from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline,RobertaModel
# from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline,RobertaTokenizer

# def load_vocab(vocab_file):
#     """Loads a vocabulary file into a dictionary."""
#     vocab = collections.OrderedDict()
#     f = open(vocab_file,)
#     reader = json.load(f) 
#     for token in reader.keys():
#         index = reader[token]
#         token = token.encode("ascii", "ignore").decode()
#         token = ''.join(token.split())
#         #print("token: '",token,"'")
#         vocab[index] = token
#     f.close()
#     return vocab

# # print(12331)
# GRAPH_CODEBERT_MLM="pre-trained/graphcodebert-base"
# tokenizer = AutoTokenizer.from_pretrained(GRAPH_CODEBERT_MLM)
# model = AutoModelForMaskedLM.from_pretrained(GRAPH_CODEBERT_MLM)
# # config = RobertaConfig.from_pretrained(CODEBERT_MLM)

# # encoder_ast = RobertaModel(config)
# # encoder_mask = RobertaModel(config)
# # decoder = RobertaForMaskedLM

# # CODE = '''void addMessageContext(MessageContext mc) throws AxisFault {
# #         mc.setServiceContext(sc);
# #         if (mc.<mask>() == null) {
# #             setMessageID(mc);
# #         }
# #         axisOp.registerOperationContext(mc, oc);
# #     }'''
# # CODE = "Paris is the <mask> of France."
# CODE = sys.argv[1]

# # Load vocab file
# my_dict = load_vocab('pre-trained/graphcodebert-base/vocab.json')

# tokenized_CODE = tokenizer.tokenize(CODE)
# # print(tokenized_CODE)
# tokens_count = str(len(tokenized_CODE))

# mask_index = tokenized_CODE.index("<mask>")
# start_index = max(0, mask_index - 255)
# stop_index = min(len(tokenized_CODE), mask_index + 255) - 1
# SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,509)]
# #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
# tokens_ids = tokenizer.convert_tokens_to_ids(SHRINKED_TOKENS)
# SHRINKED_CODE = tokenizer.decode(tokens_ids)

# SHRINKED_CODE_JSON = "{'masked_seq':" + repr(SHRINKED_CODE) + "}"
# print(SHRINKED_CODE_JSON)

# fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
# outputs = fill_mask(SHRINKED_CODE)


# for out in outputs:
# 	json_str = json.dumps(out)
# 	json_object = json.loads(json_str)
# 	token_str_exists = False; #"token_str" in json_object
# 	if not token_str_exists:
# 		index = json_object["token"]
# 		token_str = my_dict[index] 
# 		token_str = token_str.encode("ascii", "ignore").decode()
# 		json_object['token_str'] = token_str
# 	print(json_object)

  

  
