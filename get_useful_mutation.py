import pandas as pd
import os
from get_token_list import get_token_list

def do_mutation(project,version,file_name,file_path,threshold):
    # 针对传入的文件名执行变异
    in_path = f"/data02/mary/Mary/dataset_source/{project}_{version}/{file_path}"
    out_path = f"/data02/mary/mBERT-main/data/{project}_{version}/{file_name}"
    os.system(f"rm -rf {out_path}")
    os.system(f"sudo /data02/mary/mBERT-main/mBERT.sh -in={in_path} -out={out_path} -N=100000")

    # 统计codebert生成评分在threshold之上的才使用
    df = pd.read_csv(f"/data02/mary/mBERT-main/data/{project}_{version}/{file_name}/{file_name}-mapping.csv")
    useful_index = []
    for row in df.itertuples():
        if row.pred_score >= threshold:
            useful_index.append(row.id)

    # 求所有有效变异的ast_token序列
    all_tokens = []
    for index in useful_index:
        file_path = out_path+"/generated/" + str(index)
        all_tokens.append(get_token_list(file_path))
    print(len(all_tokens))


do_mutation("ant","1.5","Main","org/apache/tools/ant/Main.java",0.5)








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