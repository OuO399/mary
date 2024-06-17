from flask import Flask,request
from flask import jsonify
import sys
import os
import json
import javalang
from load_model import *
import torch

os.environ['TORCH_USE_CUDA_DSA'] = '1'



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

app = Flask(__name__)

# app.config['PROPAGATE_EXCEPTIONS'] = False

@app.route('/graphc_use_ast1',methods=["POST"])
def maskedLM_graphc1():
    data = json.loads(request.get_data())['data']

    CODE = data["CODE"]
    method_code = data["method_code"]


    # method_ast_token = get_ast_tokens(method_code)

    tokenized_CODE = tokenizer_graphc1.tokenize(CODE)
    tokens_count = str(len(tokenized_CODE))

    mask_index = tokenized_CODE.index("<mask>")
    start_index = max(0, mask_index - 200)
    stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
    SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
    #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
    tokens_ids = tokenizer_graphc1.convert_tokens_to_ids(SHRINKED_TOKENS)
    SHRINKED_CODE = tokenizer_graphc1.decode(tokens_ids)
    # # input = f'{method_ast_token} {SHRINKED_CODE}'
    input = SHRINKED_CODE
    outputs = fill_mask_graphc1(input)
    return{"data":outputs,
           "msg":"获取成功",
           "success":1}

@app.route('/graphc_use_ast2',methods=["POST"])
def maskedLM_graphc2():
    data = json.loads(request.get_data())['data']

    CODE = data["CODE"]
    method_code = data["method_code"]


    # method_ast_token = get_ast_tokens(method_code)

    tokenized_CODE = tokenizer_graphc2.tokenize(CODE)
    tokens_count = str(len(tokenized_CODE))

    mask_index = tokenized_CODE.index("<mask>")
    start_index = max(0, mask_index - 200)
    stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
    SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
    #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
    tokens_ids = tokenizer_graphc2.convert_tokens_to_ids(SHRINKED_TOKENS)
    SHRINKED_CODE = tokenizer_graphc2.decode(tokens_ids)
    # input = f'{method_ast_token} {SHRINKED_CODE}'
    input = SHRINKED_CODE
    outputs = fill_mask_graphc2(input)
    return{"data":outputs,
           "msg":"获取成功",
           "success":1}

@app.route('/graphc_use_ast3',methods=["POST"])
def maskedLM_graphc3():
    data = json.loads(request.get_data())['data']

    CODE = data["CODE"]
    method_code = data["method_code"]


    # method_ast_token = get_ast_tokens(method_code)

    tokenized_CODE = tokenizer_graphc3.tokenize(CODE)
    tokens_count = str(len(tokenized_CODE))

    mask_index = tokenized_CODE.index("<mask>")
    start_index = max(0, mask_index - 200)
    stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
    SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
    #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
    tokens_ids = tokenizer_graphc3.convert_tokens_to_ids(SHRINKED_TOKENS)
    SHRINKED_CODE = tokenizer_graphc3.decode(tokens_ids)
    # input = f'{method_ast_token} {SHRINKED_CODE}'
    input = SHRINKED_CODE
    outputs = fill_mask_graphc3(input)
    return{"data":outputs,
           "msg":"获取成功",
           "success":1}

@app.route('/graphc_use_ast4',methods=["POST"])
def maskedLM_graphc4():
    data = json.loads(request.get_data())['data']

    CODE = data["CODE"]
    method_code = data["method_code"]


    # method_ast_token = get_ast_tokens(method_code)

    tokenized_CODE = tokenizer_graphc4.tokenize(CODE)
    tokens_count = str(len(tokenized_CODE))

    mask_index = tokenized_CODE.index("<mask>")
    start_index = max(0, mask_index - 200)
    stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
    SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
    #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
    tokens_ids = tokenizer_graphc4.convert_tokens_to_ids(SHRINKED_TOKENS)
    SHRINKED_CODE = tokenizer_graphc4.decode(tokens_ids)
    # input = f'{method_ast_token} {SHRINKED_CODE}'
    input = SHRINKED_CODE
    outputs = fill_mask_graphc4(input)
    return{"data":outputs,
           "msg":"获取成功",
           "success":1}

@app.route('/graphc_use_ast5',methods=["POST"])
def maskedLM_graphc5():
    data = json.loads(request.get_data())['data']

    CODE = data["CODE"]
    method_code = data["method_code"]


    # method_ast_token = get_ast_tokens(method_code)

    tokenized_CODE = tokenizer_graphc5.tokenize(CODE)
    tokens_count = str(len(tokenized_CODE))

    mask_index = tokenized_CODE.index("<mask>")
    start_index = max(0, mask_index - 200)
    stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
    SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
    #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
    tokens_ids = tokenizer_graphc5.convert_tokens_to_ids(SHRINKED_TOKENS)
    SHRINKED_CODE = tokenizer_graphc5.decode(tokens_ids)
    # input = f'{method_ast_token} {SHRINKED_CODE}'
    input = SHRINKED_CODE
    outputs = fill_mask_graphc5(input)
    return{"data":outputs,
           "msg":"获取成功",
           "success":1}

@app.route('/graphc_use_ast6',methods=["POST"])
def maskedLM_graphc6():
    data = json.loads(request.get_data())['data']

    CODE = data["CODE"]
    method_code = data["method_code"]


    # method_ast_token = get_ast_tokens(method_code)

    tokenized_CODE = tokenizer_graphc6.tokenize(CODE)
    tokens_count = str(len(tokenized_CODE))

    mask_index = tokenized_CODE.index("<mask>")
    start_index = max(0, mask_index - 200)
    stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
    SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
    #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
    tokens_ids = tokenizer_graphc6.convert_tokens_to_ids(SHRINKED_TOKENS)
    SHRINKED_CODE = tokenizer_graphc6.decode(tokens_ids)
    # input = f'{method_ast_token} {SHRINKED_CODE}'
    input = SHRINKED_CODE
    outputs = fill_mask_graphc6(input)
    return{"data":outputs,
           "msg":"获取成功",
           "success":1}

@app.route('/graphc_use_ast7',methods=["POST"])
def maskedLM_graphc7():
    data = json.loads(request.get_data())['data']

    CODE = data["CODE"]
    method_code = data["method_code"]


    # method_ast_token = get_ast_tokens(method_code)

    tokenized_CODE = tokenizer_graphc7.tokenize(CODE)
    tokens_count = str(len(tokenized_CODE))

    mask_index = tokenized_CODE.index("<mask>")
    start_index = max(0, mask_index - 200)
    stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
    SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
    #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
    tokens_ids = tokenizer_graphc7.convert_tokens_to_ids(SHRINKED_TOKENS)
    SHRINKED_CODE = tokenizer_graphc7.decode(tokens_ids)
    # input = f'{method_ast_token} {SHRINKED_CODE}'
    input = SHRINKED_CODE
    outputs = fill_mask_graphc7(input)
    return{"data":outputs,
           "msg":"获取成功",
           "success":1}

@app.route('/graphc_use_ast8',methods=["POST"])
def maskedLM_graphc8():
    data = json.loads(request.get_data())['data']

    CODE = data["CODE"]
    method_code = data["method_code"]


    # method_ast_token = get_ast_tokens(method_code)

    tokenized_CODE = tokenizer_graphc8.tokenize(CODE)
    tokens_count = str(len(tokenized_CODE))

    mask_index = tokenized_CODE.index("<mask>")
    start_index = max(0, mask_index - 200)
    stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
    SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
    #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
    tokens_ids = tokenizer_graphc8.convert_tokens_to_ids(SHRINKED_TOKENS)
    SHRINKED_CODE = tokenizer_graphc8.decode(tokens_ids)
    # input = f'{method_ast_token} {SHRINKED_CODE}'
    input = SHRINKED_CODE
    outputs = fill_mask_graphc8(input)
    return{"data":outputs,
           "msg":"获取成功",
           "success":1}

@app.route('/graphc_use_ast9',methods=["POST"])
def maskedLM_graphc9():
    data = json.loads(request.get_data())['data']

    CODE = data["CODE"]
    method_code = data["method_code"]


    # method_ast_token = get_ast_tokens(method_code)

    tokenized_CODE = tokenizer_graphc9.tokenize(CODE)
    tokens_count = str(len(tokenized_CODE))

    mask_index = tokenized_CODE.index("<mask>")
    start_index = max(0, mask_index - 200)
    stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
    SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
    #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
    tokens_ids = tokenizer_graphc9.convert_tokens_to_ids(SHRINKED_TOKENS)
    SHRINKED_CODE = tokenizer_graphc9.decode(tokens_ids)
    # input = f'{method_ast_token} {SHRINKED_CODE}'
    input = SHRINKED_CODE
    outputs = fill_mask_graphc9(input)
        
    return{"data":outputs,
           "msg":"获取成功",
           "success":1}

@app.route('/graphc_use_ast0',methods=["POST"])
def maskedLM_graphc10():
    data = json.loads(request.get_data())['data']

    CODE = data["CODE"]
    method_code = data["method_code"]


    # method_ast_token = get_ast_tokens(method_code)

    tokenized_CODE = tokenizer_graphc10.tokenize(CODE)
    tokens_count = str(len(tokenized_CODE))

    mask_index = tokenized_CODE.index("<mask>")
    start_index = max(0, mask_index - 200)
    stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
    SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
    #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
    tokens_ids = tokenizer_graphc10.convert_tokens_to_ids(SHRINKED_TOKENS)
    SHRINKED_CODE = tokenizer_graphc10.decode(tokens_ids)
    # input = f'{method_ast_token} {SHRINKED_CODE}'
    input = SHRINKED_CODE
    outputs = fill_mask_graphc10(input)
    return{"data":outputs,
           "msg":"获取成功",
           "success":1}

# @app.route('/graphc_use_ast11',methods=["POST"])
# def maskedLM_graphc11():
#     data = json.loads(request.get_data())['data']

#     CODE = data["CODE"]
#     method_code = data["method_code"]


#     # method_ast_token = get_ast_tokens(method_code)

#     tokenized_CODE = tokenizer_graphc11.tokenize(CODE)
#     tokens_count = str(len(tokenized_CODE))

#     mask_index = tokenized_CODE.index("<mask>")
#     start_index = max(0, mask_index - 200)
#     stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
#     SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
#     #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
#     tokens_ids = tokenizer_graphc11.convert_tokens_to_ids(SHRINKED_TOKENS)
#     SHRINKED_CODE = tokenizer_graphc11.decode(tokens_ids)
#     # input = f'{method_ast_token} {SHRINKED_CODE}'
#     input = SHRINKED_CODE
#     outputs = fill_mask_graphc11(input)
#     return{"data":outputs,
#            "msg":"获取成功",
#            "success":1}

# @app.route('/graphc_use_ast12',methods=["POST"])
# def maskedLM_graphc12():
#     data = json.loads(request.get_data())['data']

#     CODE = data["CODE"]
#     method_code = data["method_code"]


#     # method_ast_token = get_ast_tokens(method_code)

#     tokenized_CODE = tokenizer_graphc12.tokenize(CODE)
#     tokens_count = str(len(tokenized_CODE))

#     mask_index = tokenized_CODE.index("<mask>")
#     start_index = max(0, mask_index - 200)
#     stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
#     SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
#     #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
#     tokens_ids = tokenizer_graphc12.convert_tokens_to_ids(SHRINKED_TOKENS)
#     SHRINKED_CODE = tokenizer_graphc12.decode(tokens_ids)
#     # input = f'{method_ast_token} {SHRINKED_CODE}'
#     input = SHRINKED_CODE
#     outputs = fill_mask_graphc12(input)
#     return{"data":outputs,
#            "msg":"获取成功",
#            "success":1}

# @app.route('/graphc_use_ast13',methods=["POST"])
# def maskedLM_graphc13():
#     data = json.loads(request.get_data())['data']

#     CODE = data["CODE"]
#     method_code = data["method_code"]


#     # method_ast_token = get_ast_tokens(method_code)

#     tokenized_CODE = tokenizer_graphc13.tokenize(CODE)
#     tokens_count = str(len(tokenized_CODE))

#     mask_index = tokenized_CODE.index("<mask>")
#     start_index = max(0, mask_index - 200)
#     stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
#     SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
#     #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
#     tokens_ids = tokenizer_graphc13.convert_tokens_to_ids(SHRINKED_TOKENS)
#     SHRINKED_CODE = tokenizer_graphc13.decode(tokens_ids)
#     # input = f'{method_ast_token} {SHRINKED_CODE}'
#     input = SHRINKED_CODE
#     outputs = fill_mask_graphc13(input)
#     return{"data":outputs,
#            "msg":"获取成功",
#            "success":1}

# @app.route('/graphc_use_ast14',methods=["POST"])
# def maskedLM_graphc14():
#     data = json.loads(request.get_data())['data']

#     CODE = data["CODE"]
#     method_code = data["method_code"]


#     # method_ast_token = get_ast_tokens(method_code)

#     tokenized_CODE = tokenizer_graphc14.tokenize(CODE)
#     tokens_count = str(len(tokenized_CODE))

#     mask_index = tokenized_CODE.index("<mask>")
#     start_index = max(0, mask_index - 200)
#     stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
#     SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
#     #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
#     tokens_ids = tokenizer_graphc14.convert_tokens_to_ids(SHRINKED_TOKENS)
#     SHRINKED_CODE = tokenizer_graphc14.decode(tokens_ids)
#     # input = f'{method_ast_token} {SHRINKED_CODE}'
#     input = SHRINKED_CODE
#     outputs = fill_mask_graphc14(input)
#     return{"data":outputs,
#            "msg":"获取成功",
#            "success":1}

# @app.route('/graphc_use_ast15',methods=["POST"])
# def maskedLM_graphc15():
#     data = json.loads(request.get_data())['data']

#     CODE = data["CODE"]
#     method_code = data["method_code"]


#     # method_ast_token = get_ast_tokens(method_code)

#     tokenized_CODE = tokenizer_graphc15.tokenize(CODE)
#     tokens_count = str(len(tokenized_CODE))

#     mask_index = tokenized_CODE.index("<mask>")
#     start_index = max(0, mask_index - 200)
#     stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
#     SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
#     #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
#     tokens_ids = tokenizer_graphc15.convert_tokens_to_ids(SHRINKED_TOKENS)
#     SHRINKED_CODE = tokenizer_graphc15.decode(tokens_ids)
#     # input = f'{method_ast_token} {SHRINKED_CODE}'
#     input = SHRINKED_CODE
#     outputs = fill_mask_graphc15(input)
#     return{"data":outputs,
#            "msg":"获取成功",
#            "success":1}

# @app.route('/graphc_use_ast16',methods=["POST"])
# def maskedLM_graphc16():
#     data = json.loads(request.get_data())['data']

#     CODE = data["CODE"]
#     method_code = data["method_code"]


#     # method_ast_token = get_ast_tokens(method_code)

#     tokenized_CODE = tokenizer_graphc16.tokenize(CODE)
#     tokens_count = str(len(tokenized_CODE))

#     mask_index = tokenized_CODE.index("<mask>")
#     start_index = max(0, mask_index - 200)
#     stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
#     SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
#     #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
#     tokens_ids = tokenizer_graphc16.convert_tokens_to_ids(SHRINKED_TOKENS)
#     SHRINKED_CODE = tokenizer_graphc16.decode(tokens_ids)
#     # input = f'{method_ast_token} {SHRINKED_CODE}'
#     input = SHRINKED_CODE
#     outputs = fill_mask_graphc16(input)
#     return{"data":outputs,
#            "msg":"获取成功",
#            "success":1}

# @app.route('/graphc_use_ast17',methods=["POST"])
# def maskedLM_graphc17():
#     data = json.loads(request.get_data())['data']

#     CODE = data["CODE"]
#     method_code = data["method_code"]


#     # method_ast_token = get_ast_tokens(method_code)

#     tokenized_CODE = tokenizer_graphc17.tokenize(CODE)
#     tokens_count = str(len(tokenized_CODE))

#     mask_index = tokenized_CODE.index("<mask>")
#     start_index = max(0, mask_index - 200)
#     stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
#     SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
#     #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
#     tokens_ids = tokenizer_graphc17.convert_tokens_to_ids(SHRINKED_TOKENS)
#     SHRINKED_CODE = tokenizer_graphc17.decode(tokens_ids)
#     # input = f'{method_ast_token} {SHRINKED_CODE}'
#     input = SHRINKED_CODE
#     outputs = fill_mask_graphc17(input)
#     return{"data":outputs,
#            "msg":"获取成功",
#            "success":1}

# @app.route('/graphc_use_ast18',methods=["POST"])
# def maskedLM_graphc18():
#     data = json.loads(request.get_data())['data']

#     CODE = data["CODE"]
#     method_code = data["method_code"]


#     # method_ast_token = get_ast_tokens(method_code)

#     tokenized_CODE = tokenizer_graphc18.tokenize(CODE)
#     tokens_count = str(len(tokenized_CODE))

#     mask_index = tokenized_CODE.index("<mask>")
#     start_index = max(0, mask_index - 200)
#     stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
#     SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
#     #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
#     tokens_ids = tokenizer_graphc18.convert_tokens_to_ids(SHRINKED_TOKENS)
#     SHRINKED_CODE = tokenizer_graphc18.decode(tokens_ids)
#     # input = f'{method_ast_token} {SHRINKED_CODE}'
#     input = SHRINKED_CODE
#     outputs = fill_mask_graphc18(input)
#     return{"data":outputs,
#            "msg":"获取成功",
#            "success":1}

# @app.route('/graphc_use_ast19',methods=["POST"])
# def maskedLM_graphc19():
#     data = json.loads(request.get_data())['data']

#     CODE = data["CODE"]
#     method_code = data["method_code"]


#     # method_ast_token = get_ast_tokens(method_code)

#     tokenized_CODE = tokenizer_graphc19.tokenize(CODE)
#     tokens_count = str(len(tokenized_CODE))

#     mask_index = tokenized_CODE.index("<mask>")
#     start_index = max(0, mask_index - 200)
#     stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
#     SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
#     #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
#     tokens_ids = tokenizer_graphc19.convert_tokens_to_ids(SHRINKED_TOKENS)
#     SHRINKED_CODE = tokenizer_graphc19.decode(tokens_ids)
#     # input = f'{method_ast_token} {SHRINKED_CODE}'
#     input = SHRINKED_CODE
#     outputs = fill_mask_graphc19(input)
#     return{"data":outputs,
#            "msg":"获取成功",
#            "success":1}

# @app.route('/graphc_use_ast0',methods=["POST"])
# def maskedLM_graphc0():
#     data = json.loads(request.get_data())['data']

#     CODE = data["CODE"]
#     method_code = data["method_code"]


#     # method_ast_token = get_ast_tokens(method_code)

#     tokenized_CODE = tokenizer_graphc20.tokenize(CODE)
#     tokens_count = str(len(tokenized_CODE))

#     mask_index = tokenized_CODE.index("<mask>")
#     start_index = max(0, mask_index - 200)
#     stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
#     SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
#     #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
#     tokens_ids = tokenizer_graphc20.convert_tokens_to_ids(SHRINKED_TOKENS)
#     SHRINKED_CODE = tokenizer_graphc20.decode(tokens_ids)
#     # input = f'{method_ast_token} {SHRINKED_CODE}'
#     input = SHRINKED_CODE
#     outputs = fill_mask_graphc20(input)
#     return{"data":outputs,
#            "msg":"获取成功",
#            "success":1}




# @app.route('/codebert',methods=["POST"])
# def maskedLM_codebert():
#     data = json.loads(request.get_data())['data']

#     CODE = data["CODE"]
#     method_code = data["method_code"]


#     # method_ast_token = get_ast_tokens(method_code)

#     tokenized_CODE = tokenizer_codebert.tokenize(CODE)
#     tokens_count = str(len(tokenized_CODE))

#     mask_index = tokenized_CODE.index("<mask>")
#     start_index = max(0, mask_index - 200)
#     stop_index = min(len(tokenized_CODE), mask_index + 200) - 1
#     SHRINKED_TOKENS = tokenized_CODE[start_index:max(stop_index,400)]
#     #assert(len(SHRINKED_TOKENS)) #512 is the maximum sequence length for codebert model
#     tokens_ids = tokenizer_codebert.convert_tokens_to_ids(SHRINKED_TOKENS)
#     SHRINKED_CODE = tokenizer_codebert.decode(tokens_ids)
#     # input = f'{method_ast_token} {SHRINKED_CODE}'
    # input = SHRINKED_CODE
#     outputs = fill_mask_codebert(input)
#     return{"data":outputs,
#            "msg":"获取成功",
#            "success":1}



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8088,debug=True)