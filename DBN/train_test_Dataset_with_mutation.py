
import pickle
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from dbn import UnsupervisedDBN, AbstractSupervisedDBN, NumPyAbstractSupervisedDBN, SupervisedDBNClassification
import pandas as pd
import os
from copy import deepcopy
import random


def pre_process(project_name, version,cut_length):
    df = pd.read_csv("../PROMISE/promise_data/{}/{}.csv".format(project_name, version))
    # print(df[df['bugs']==1])
    label_dict = {}
    # without_bug_num = 0
    # with_bug_num = 0
    for i in range(len(df)):
        # print(type(df.iloc[i]["name"]))
        # print(df.iloc[i]["bugs"])
        if df.iloc[i]["bugs"] != 0:
            label_dict[df.iloc[i]["name"] + ".java"] = 1
            # with_bug_num += 1
        else:
            label_dict[df.iloc[i]["name"] + ".java"] = df.iloc[i]["bugs"]
            # without_bug_num += 1
    # print(label_dict)
    with open("../numtokens/{}_{}_without_mutation.pkl".format(project_name, version), "rb") as f1:
        num_tokens_dict = pickle.load(f1)
        print(len(num_tokens_dict))
        file_names = num_tokens_dict.keys()
    # with open("../numtokens/{}_{}_with_mutation_file.pkl".format(project_name, version), "rb") as f:
    #     num_tokens_dict_with_mutation = pickle.load(f)
    #     # file_names = num_tokens_dict_with_mutation.keys()

    #     # 从变异文件的token序列中随机选取和原始数据相同大小的样本数
    #     num = without_bug_num-with_bug_num
    #     numArray = set()
    #     while len(numArray)<num:
    #         numArray.add(random.randint(0,len(num_tokens_dict_with_mutation)-1))
    #     numlist = list(numArray)
    #     num_tokens_dict = deepcopy(num_tokens_dict_without_mutation)
    #     for i in numlist:
    #         key = list(num_tokens_dict_with_mutation.keys())[i]
    #         num_tokens_dict[key] = num_tokens_dict_with_mutation[key]
    #     file_names = num_tokens_dict.keys()
    #     print(len(num_tokens_dict))

    #     # 将变异后的文件也加入label_dict中，并标记为有缺陷文件
    #     label_dict_file_name = deepcopy(list(label_dict.keys()))
    #     for i in label_dict_file_name:
    #         for j in file_names:
    #             if (i in j) and (i != j):
    #                 label_dict[j] = 1

        DBN_data = []
        labels = []
        max_length = 0
        for key, i in num_tokens_dict.items():
            if (len(i) > max_length):
                max_length = len(i)
            # if(len(i) == 7063):
            #     print(key)
        print(max_length)



        #将数据全都裁剪或填充到固定长度
        fix_length = int(max_length*cut_length)
        for i in num_tokens_dict.keys():
            if(len(num_tokens_dict[i])>fix_length):
                num_tokens_dict[i] = num_tokens_dict[i][:fix_length]
            else:
                length = len(num_tokens_dict[i])
                for _ in range(fix_length-length):
                    num_tokens_dict[i].append(0)

        for i in file_names:
            try:
                labels.append(label_dict[i])
                DBN_data.append(np.array(num_tokens_dict[i]))
            except KeyError:
                continue

        # print(np.array((DBN_data[1])).shape)
        # print(labels)
        return np.array(DBN_data), labels,str(int(cut_length*100))+"%"

def mutation_oversample(project_name,version,X,Y):
    with open("../numtokens/{}_{}_with_mutation_file.pkl".format(project_name, version), "rb") as f:
        num_tokens_dict_with_mutation = pickle.load(f)
        # file_names = num_tokens_dict_with_mutation.keys()

        # 从变异文件的token序列中随机选取和原始数据相同大小的样本数
        fix_length = len(X[0])
        print("fix_length={}".format(fix_length))
        num = np.sum(Y==0)-np.sum(Y==1)
        numArray = set()
        while len(numArray)<num:
            numArray.add(random.randint(0,len(num_tokens_dict_with_mutation)-1))
        numlist = list(numArray)
        X = list(X)
        for i in numlist:
            key = list(num_tokens_dict_with_mutation.keys())[i]
            num_tokens = num_tokens_dict_with_mutation[key]
            if(len(num_tokens)>fix_length):
                num_tokens = num_tokens[:fix_length]
            else:
                length = len(num_tokens)
                for _ in range(fix_length-length):
                    num_tokens.append(0)
            X.append(num_tokens)
            Y = np.append(Y,1)
        return np.array(X),Y


def handle_batch(project,cut_length):
    # 获取每个项目版本号
    project_path = "../PROMISE/promise_data/{}".format(project)
    versions = []
    for root, dirnames, filenames in os.walk(project_path):
        for i in filenames:
            versions.append(i[0:-4])
    # 根据项目名和版本号获取最终的数据，并使用sklearn进行训练测试集划分
    print(versions)
    for version in versions:
        # version = "4.3"
        print(version)
        DBN_data, labels,length = pre_process(project, version,cut_length)
        # print(DBN_data.shape)
        # data1 = DBN_data[:without_mutation_length]
        # labels1 = labels[:without_mutation_length]
        # data2 = DBN_data[without_mutation_length:]
        # labels2 = labels[without_mutation_length:]
        # print(len(data1))
        train_X, test_X, train_Y, test_Y = train_test_split(DBN_data, labels, test_size=0.2, random_state=66)
        trait_X_resample, train_Y_resample = mutation_oversample(project, version, train_X, np.array(train_Y))
        # train_X  += data2
        # train_Y += labels2
        # train_X = np.array(train_X)
        # test_X = np.array(test_X)

        #获取对应项目词向量表大小来进行归一化
        max_num = 1
        with open("../vocabdict/{}_{}.pkl".format(project,version),'rb') as f:
            vocabdict = pickle.load(f)
            max_num = len(vocabdict)

        # 构建dbn模型进行训练
        # temp_vector=train_X[1]
        trait_X_resample = trait_X_resample / (max_num)
        test_X = test_X / max_num
        # print(np.linalg.norm(train_X - temp_vector, axis=1))
        # exit()
        within_dbn = UnsupervisedDBN(hidden_layers_structure=[512, 256, 100],
                                     batch_size=128,
                                     learning_rate_rbm=0.06,
                                     n_epochs_rbm=100,
                                     activation_function='sigmoid',
                                     verbose=False)

        print(trait_X_resample.shape)
        within_dbn_input = trait_X_resample
        # print(train_X[0])
        # print(train_X[1])
        print('{} within DBN train started'.format(project))
        within_dbn.fit(within_dbn_input)
        # within_dbn_sup.fit(train_X,train_Y)
        print('{} within DBN train ended'.format(project))
        within_train_dbn_output = within_dbn.transform(within_dbn_input)
        print(within_train_dbn_output.shape)

        np.save('./train_data/{}_{}_train_X_with_mutation_{}.npy'.format(project,version,length), within_train_dbn_output)

        np.save('./train_data/{}_{}_train_Y_with_mutation_{}.npy'.format(project,version,length), train_Y_resample)

        within_test_dbn_output = within_dbn.transform(test_X)
        np.save('./test_data/{}_{}_test_X_with_mutation_{}.npy'.format(project,version,length), within_test_dbn_output)

        np.save('./test_data/{}_{}_test_Y_with_mutation_{}.npy'.format(project,version,length), np.array(test_Y))



# pre_process("lucene",'2.4')
projects = ['jEdit','ant','ivy']
for i in projects:
    handle_batch(i,1)
# handle_batch("ant")