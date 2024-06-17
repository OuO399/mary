#!/usr/bin/python

import sys
import json
import collections
import string
import javalang
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline,RobertaModel

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

def get_ast_tokens(method_code):
    java_code = "public class test{\n"+method_code+"}"
    tree = javalang.parse.parse(java_code)
    alltokens=[]
    for path, node in tree:
        if path==[] or path==():
            continue
        # print(type(node))
        alltokens.append(node.__class__.__name__)
    # print(alltokens)
    return alltokens


CODEBERT_MLM="pre-trained/codebert-base-mlm"
model = RobertaForMaskedLM.from_pretrained(CODEBERT_MLM)
tokenizer = RobertaTokenizer.from_pretrained(CODEBERT_MLM)
# config = RobertaConfig.from_pretrained(CODEBERT_MLM)

# encoder_ast = RobertaModel(config)
# encoder_mask = RobertaModel(config)
# decoder = RobertaForMaskedLM


CODE = sys.argv[1]
method_code = sys.argv[2]
method_ast_token = get_ast_tokens(method_code)

# Load vocab file
my_dict = load_vocab('pre-trained/codebert-base-mlm/vocab.json')

tokenized_CODE = tokenizer.tokenize(CODE)
tokens_count = str(len(tokenized_CODE))

mask_index = tokenized_CODE.index("<mask>")
start_index = max(0, mask_index - 125)
stop_index = min(len(tokenized_CODE), mask_index + 125) - 1
SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,250)]
#assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
tokens_ids = tokenizer.convert_tokens_to_ids(SHRINKED_TOKENS)
SHRINKED_CODE = tokenizer.decode(tokens_ids)
WITH_AST = f'{method_ast_token} {SHRINKED_CODE}'

SHRINKED_CODE_JSON = "{'masked_seq':" + repr(SHRINKED_CODE) + "}"
print(SHRINKED_CODE_JSON)


fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
# outputs = fill_mask(SHRINKED_CODE)
outputs = fill_mask(WITH_AST)


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

  

  
