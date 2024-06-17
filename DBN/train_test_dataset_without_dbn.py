import pickle
import numpy as np
import sklearn
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
from dbn import UnsupervisedDBN,AbstractSupervisedDBN,NumPyAbstractSupervisedDBN,SupervisedDBNClassification
import pandas as pd
import os
from copy import deepcopy
from imblearn.over_sampling import RandomOverSampler as  ROS
import threading
from multiprocessing import Process
import time
import random

def pre_process(project_name,version,cut_length):
    df = pd.read_csv("../PROMISE/promise_data/{}/{}.csv".format(project_name,version))
    # print(df[df['bugs']==1])
    label_dict = {}
    for i in range(len(df)):
        # print(type(df.iloc[i]["name"]))
        # print(df.iloc[i]["bugs"])
        if df.iloc[i]["bugs"] == 0:
            label_dict[df.iloc[i]["name"]+".java"] = 0
        else:
            label_dict[df.iloc[i]["name"]+".java"] = 1
    # print(label_dict)
    with open("../numtokens/{}_{}_without_mutation.pkl".format(project_name,version),"rb") as f:
        num_tokens_dict = pickle.load(f)
        max_length = 0
        for key,i in num_tokens_dict.items():
            if(len(i)>max_length):
                max_length = len(i)
            # if(len(i) == 7063):
            #     print(key)
        print(max_length)


        # # 将所有数据填充到最长数据的长度
        # for i in num_tokens_dict.keys():
        #     for _ in range(max_length-len(num_tokens_dict[i])):
        #         num_tokens_dict[i].append(0)

        # 将数据全都裁剪或填充到固定长度
        fix_length = int(max_length*cut_length)
        for i in num_tokens_dict.keys():
            if(len(num_tokens_dict[i])>fix_length):
                num_tokens_dict[i] = num_tokens_dict[i][:fix_length]
            else:
                length = len(num_tokens_dict[i])
                for _ in range(fix_length-length):
                    num_tokens_dict[i].append(0)

        

        # for i in file_names:
        #     try:
        #         labels.append(label_dict[i])
        #         DBN_data.append(np.array(num_tokens_dict[i]))
        #     except KeyError:
        #         continue

        # print(np.array((DBN_data[1])).shape)
        # print(labels)
        return num_tokens_dict,label_dict,str(int(cut_length*100))+"%"

def without_oversample(project,version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict):
    print("without_oversample start")
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    train_file_names = train_X_dict.keys()
    for i in train_file_names:
        train_X_list.append(train_X_dict[i])
        train_Y_list.append(train_Y_dict[i])
    train_X = np.array(train_X_list)
    train_Y = np.array(train_Y_list)
    test_file_names = test_X_dict.keys()
    for i in test_file_names:
        test_X_list.append(test_X_dict[i])
        test_Y_list.append(test_Y_dict[i])
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)

    dbn_train(project,version,"without_mutation",cut_length,train_X,train_Y,test_X,test_Y)
    print("without_oversample end")

def mutation_oversample(project_name,version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict):
    print("mutation_oversample start")
    with open("../numtokens/{}_{}_with_mutation_file.pkl".format(project_name, version), "rb") as f:
        num_tokens_dict_with_mutation = pickle.load(f)
        # file_names = num_tokens_dict_with_mutation.keys()
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    train_file_names = train_X_dict.keys()
    for i in train_file_names:
        train_X_list.append(train_X_dict[i])
        train_Y_list.append(train_Y_dict[i])
    train_Y = np.array(train_Y_list)
    test_file_names = test_X_dict.keys()
    for i in test_file_names:
        test_X_list.append(test_X_dict[i])
        test_Y_list.append(test_Y_dict[i])
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)
    print(len(train_X_list))
    print(len(train_Y))

    # 从变异文件的token序列中随机选取和原始数据相同大小的样本数
    fix_length = len(train_X_list[0])
    print("fix_length={}".format(fix_length))
    num = np.sum(train_Y==0)-np.sum(train_Y==1)
    if (len(num_tokens_dict_with_mutation)<num) :
        numlist = [i for i in range(len(num_tokens_dict_with_mutation))]
    else:
        numlist = random.sample(range(0,len(num_tokens_dict_with_mutation)-1),num)
        # while len(numArray)<num:
        #     random_num = random.randint(0,len(num_tokens_dict_with_mutation)-1)
        #     # 确定变异文件为训练集中存在的文件变异而来的
        #     # 工具生成的变异文件
        #     file_name = list(num_tokens_dict_with_mutation.keys())[random_num].split("_")[0]
        #     if(file_name in train_X_dict.keys()):
        #         numArray.add(random_num)
        # numlist = list(numArray)
    for i in numlist:
        key = list(num_tokens_dict_with_mutation.keys())[i]
        num_tokens = num_tokens_dict_with_mutation[key]
        if(len(num_tokens)>fix_length):
            num_tokens = num_tokens[:fix_length]
        else:
            length = len(num_tokens)
            for _ in range(fix_length-length):
                num_tokens.append(0)
        train_X_list.append(num_tokens)
        train_Y = np.append(train_Y,1)
    print(len(train_X_list))
    print(len(train_Y))
    train_X = np.array(train_X_list)
    dbn_train(project_name,version,"with_mutation",cut_length,train_X,train_Y,test_X,test_Y)
    print("mutation_oversample end")


def manual_mutation_oversample(project_name,version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict):
    print("manual_mutation_oversample start")
    with open("../numtokens/{}_{}_with_manual_mutation_file.pkl".format(project_name, version), "rb") as f:
        num_tokens_dict_with_mutation = pickle.load(f)
        # file_names = num_tokens_dict_with_mutation.keys()
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    train_file_names = train_X_dict.keys()
    for i in train_file_names:
        train_X_list.append(train_X_dict[i])
        train_Y_list.append(train_Y_dict[i])
    train_Y = np.array(train_Y_list)
    test_file_names = test_X_dict.keys()
    for i in test_file_names:
        test_X_list.append(test_X_dict[i])
        test_Y_list.append(test_Y_dict[i])
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)

    # 从变异文件的token序列中随机选取和原始数据相同大小的样本数
    fix_length = len(train_X_list[0])
    print("fix_length={}".format(fix_length))
    num = np.sum(train_Y==0)-np.sum(train_Y==1)
    print("num:{}".format(num))
    print("token_list_length:{}".format(len(num_tokens_dict_with_mutation)))
    if (len(num_tokens_dict_with_mutation)<=num) :
        numlist = [i for i in range(len(num_tokens_dict_with_mutation))]
    else:
        numlist = random.sample(range(0,len(num_tokens_dict_with_mutation)-1),num)
        # while len(numArray)<num:
        #     random_num = random.randint(0,len(num_tokens_dict_with_mutation)-1)
        #     # 确定变异文件为训练集中存在的文件变异而来的
        #     # 手工生成的变异文件
        #     if(list(num_tokens_dict_with_mutation.keys())[random_num] in train_X_dict.keys()):
        #         numArray.add(random_num)
        # numlist = list(numArray)
    for i in numlist:
        key = list(num_tokens_dict_with_mutation.keys())[i]
        num_tokens = num_tokens_dict_with_mutation[key]
        if(len(num_tokens)>fix_length):
            num_tokens = num_tokens[:fix_length]
        else:
            length = len(num_tokens)
            for _ in range(fix_length-length):
                num_tokens.append(0)
        train_X_list.append(num_tokens)
        train_Y = np.append(train_Y,1)
    train_X = np.array(train_X_list)
    dbn_train(project_name,version,"with_manual_mutation",cut_length,train_X,train_Y,test_X,test_Y)
    print("manual_mutation_oversample end")

def ROS_oversample(project,version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict):
    print("ROS_oversample start")
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    train_file_names = train_X_dict.keys()
    for i in train_file_names:
        train_X_list.append(train_X_dict[i])
        train_Y_list.append(train_Y_dict[i])
    train_X = np.array(train_X_list)
    train_Y = np.array(train_Y_list)
    test_file_names = test_X_dict.keys()
    for i in test_file_names:
        test_X_list.append(test_X_dict[i])
        test_Y_list.append(test_Y_dict[i])
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)

    ros = ROS(random_state=66)
    train_X_resample,train_Y_resample = ros.fit_resample(train_X,train_Y)
    dbn_train(project,version,"with_ROS",cut_length,train_X_resample,train_Y_resample,test_X,test_Y)
    print("ROS_oversample end")

def dbn_train(project,version,data_type,cut_length,train_X,train_Y,test_X,test_Y):
    # 构建dbn模型进行训练
    max_num = 0
    with open("../vocabdict/{}_{}.pkl".format(project,version),'rb') as f:
        vocabdict = pickle.load(f)
        max_num = len(vocabdict)
    train_X = train_X / (max_num)
    test_X = test_X / max_num

    np.save('./train_data_without_dbn/{}_{}_train_X_{}_{}.npy'.format(project,version,data_type,cut_length), np.array(train_X))

    np.save('./train_data_without_dbn/{}_{}_train_Y_{}_{}.npy'.format(project,version,data_type,cut_length), np.array(train_Y))

    np.save('./test_data_without_dbn/{}_{}_test_X_{}_{}.npy'.format(project,version,data_type,cut_length), np.array(test_X))

    np.save('./test_data_without_dbn/{}_{}_test_Y_{}_{}.npy'.format(project,version,data_type,cut_length), np.array(test_Y))

def train_test_split(project,version,data_dict,labels_dict,test_size,random_state):
    file_num = len(labels_dict)
    if (os.path.exists("../train_test_index/{}_{}_{}_{}.pkl".format(project,version,test_size,random_state))):
        with open("../train_test_index/{}_{}_{}_{}.pkl".format(project,version,test_size,random_state),"rb")as f:
            test_index_list = pickle.load(f)
    else:
        # 随机选取指定大小的下标列表，下标对应的文件作为测试集
        # test_index = set()
        test_file_num = int(file_num * test_size)
        test_index_list = random.sample(range(0,file_num),test_file_num)
        # while len(test_index) < test_file_num:
        #     random_num = random.randint(0, file_num - 1)
        #     test_index.add(random_num)
        #     print(len(test_index))
        # test_index_list = list(test_index)
        with open("../train_test_index/{}_{}_{}_{}.pkl".format(project, version, test_size, random_state), "wb") as f:
            pickle.dump(test_index_list,f)
    train_X = {}
    train_Y = {}
    test_X = {}
    test_Y = {}
    for i in test_index_list:
        file_names = list(labels_dict.keys())
        try:
            test_X[file_names[i]] = data_dict[file_names[i]]
            test_Y[file_names[i]] = labels_dict[file_names[i]]
        except KeyError:
            continue
    for i in range(file_num):
        file_names = list(labels_dict.keys())
        if i not in test_index_list:
            try:
                train_X[file_names[i]] = data_dict[file_names[i]]
                train_Y[file_names[i]] = labels_dict[file_names[i]]
            except KeyError:
                continue
    print("train_X length:{}".format(len(train_X)))
    print("train_Y length:{}".format(len(train_Y)))
    print("test_X length:{}".format(len(test_X)))
    print("test_Y length:{}".format(len(test_Y)))
    return train_X,test_X,train_Y,test_Y




def handle_batch(project,cut_length):
    # 获取每个项目版本号
    project_path = "../PROMISE/promise_data/{}".format(project)
    versions = []
    for root,dirnames,filenames in os.walk(project_path):
        for i in filenames:
            versions.append(i[0:-4])
    # 根据项目名和版本号获取最终的数据，并使用sklearn进行训练测试集划分
    Processes = []
    for version in versions:
        print(version)
        DBN_data_dict,labels_dict,cut_length_string = pre_process(project,version,cut_length)
        # print(DBN_data.shape)
        train_X,test_X,train_Y,test_Y = train_test_split(project,version,DBN_data_dict,labels_dict,test_size=0.3,random_state = 1)


        #逐个项目跑
        # p1 = Process(target=without_oversample,args=(project,version,cut_length_string,train_X,train_Y,test_X,test_Y))
        p2 = Process(target=mutation_oversample,args=(project,version,cut_length_string,train_X,train_Y,test_X,test_Y))
        # p3 = Process(target=ROS_oversample,args=(project,version,cut_length_string,train_X,train_Y,test_X,test_Y))
        # p4 = Process(target=manual_mutation_oversample,args=(project,version,cut_length_string,train_X,train_Y,test_X,test_Y))
        start_time = time.time()
        # p1.start()
        p2.start()
        # p3.start()
        # p4.start()
        # p1.join()
        p2.join()
        # p3.join()
        # p4.join()
        end_time = time.time()
        print("time:{}".format(end_time-start_time))

    #     #所有项目一起跑
    #     Processes.append(Process(target=without_oversample,args=(project,version,cut_length,train_X,train_Y,test_X,test_Y)))
    #     Processes.append(Process(target=mutation_oversample,args=(project,version,cut_length,train_X,train_Y,test_X,test_Y)))
    #     Processes.append(Process(target=ROS_oversample,args=(project,version,cut_length,train_X,train_Y,test_X,test_Y)))
    #     Processes.append(Process(target=manual_mutation_oversample,args=(project,version,cut_length,train_X,train_Y,test_X,test_Y)))
    # start_time = time.time()
    # for process in Processes:
    #     process.start()
    # for process in Processes:
    #     process.join()
    # end_time = time.time()
    # print("time:{}".format(end_time-start_time))


# pre_process("lucene",'2.4')
# projects = ['jEdit','ant','ivy']
projects = ['jEdit']
# projects = ['ant']s
for i in projects:
    handle_batch(i,1)
# handle_batch("ant")