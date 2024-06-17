import pandas as pd
import os
from get_token_list import get_token_list
import multiprocessing

def do_mutation(args,N=100000):
    project, version, file_name, file_path, isUsedAST, threshold = args
    all_tokens = []
    process_label = int(multiprocessing.current_process().name.split("-")[1]) % 10
    # 针对传入的文件名执行变异
    in_path = f"/data02/mary/Mary/dataset_source/{project}_{version}/{file_path}"
    out_path = f"/data02/mary/mBERT-main/data_graphc_{isUsedAST}_{threshold}/{project}_{version}/{file_name}"
    new_path = "/data02/mary/mBERT-main"
    os.chdir(new_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    os.system(f"echo LABlab645 | sudo -S chmod 777 -R {out_path}/")
    os.system(f"rm -rf {out_path}/")
    result = os.system(f"echo LABlab645 | sudo -S /data02/mary/mBERT-main/mBERT.sh -in={in_path} -out={out_path} -N={N} -p={process_label}")
    print(result)
    if result == 0:
        # 统计codebert生成评分在threshold之上的才使用
        df = pd.read_csv(f"{out_path}/{file_name}-mapping.csv")
        useful_index = []
        for row in df.itertuples():
            if row.pred_score >= threshold:
                useful_index.append(row.id)

        # 求所有有效变异的ast_token序列
        for index in useful_index:
            res_file_path = out_path+"/generated/" + str(index) + "/" + file_name + ".java"
            all_tokens.append(get_token_list(res_file_path))
        print(len(all_tokens))
        print(useful_index)
    return all_tokens

def do_mutation_after_μbert(args,N=100000):
    try:
        project, version, file_name, file_path, isUsedAST, threshold = args
        all_tokens = []
        in_path = f"/data02/mary/Mary/dataset_source/{project}_{version}/{file_path}"
        out_path = f"/data02/mary/mBERT-main/data_graphc_{isUsedAST}_{threshold}/{project}_{version}/{file_name}"
        new_path = "/data02/mary/mBERT-main"
        os.chdir(new_path)
        # 统计codebert生成评分在threshold之上的才使用
        df = pd.read_csv(f"{out_path}/{file_name}-mapping.csv")
        useful_index = []
        for row in df.itertuples():
            if row.pred_score >= threshold:
                useful_index.append(row.id)

        # 求所有有效变异的ast_token序列
        for index in useful_index:
            res_file_path = out_path+"/generated/" + str(index) + "/" + file_name + ".java"
            all_tokens.append(get_token_list(res_file_path))
        print(len(all_tokens))
        print(useful_index)
        return all_tokens
    except FileNotFoundError:
        return []
# do_mutation("ant","1.5","Main","org/apache/tools/ant/Main.java",0.9,5)
# do_mutation("ant","1.5","BZip2Constants","org/apache/tools/bzip2/BZip2Constants.java",0.9)
# print(do_mutation("synapse","1.0","SynapseModule","/org/apache/synapse/core/axis2/SynapseModule.java",0.9))








# df = pd.read_csv("/data02/mary/mBERT-main/test/SimpleMethods-mapping.csv")
# out_path = "/data02/mary/mBERT-main"
# list = [1,2,3,4]
# for i in list:
#     file_path = out_path + "/123/" + str(i)
#     print(file_path)

# count = 0
# for row in df.itertuples():
#     if row.pred_score >=0.5:
#         count += 1
#     # print(row.pred_score)
    # exit()
# print(df["id"])
# print(count)
# os.system("sudo ./mBERT.sh -in=./examples/SimpleMethods.java -out=./new_result_save/test/ -N=100000")