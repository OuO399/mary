import pickle
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
from dbn import UnsupervisedDBN,AbstractSupervisedDBN,NumPyAbstractSupervisedDBN,SupervisedDBNClassification
import pandas as pd
import os
from copy import deepcopy
from imblearn.over_sampling import RandomOverSampler as  ROS

def pre_process(project_name,version,cut_length):
    df = pd.read_csv("../PROMISE/promise_data/{}/{}.csv".format(project_name,version))
    # print(df[df['bugs']==1])
    label_dict = {}
    for i in range(len(df)):
        # print(type(df.iloc[i]["name"]))
        # print(df.iloc[i]["bugs"])
        label_dict[df.iloc[i]["name"]+".java"] = df.iloc[i]["bugs"]
    # print(label_dict)
    with open("../numtokens/{}_{}_without_mutation.pkl".format(project_name,version),"rb") as f:
        num_tokens_dict = pickle.load(f)
        file_names = num_tokens_dict.keys()
        DBN_data = []
        labels = []
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

        

        for i in file_names:
            try:
                labels.append(label_dict[i])
                DBN_data.append(np.array(num_tokens_dict[i]))
            except KeyError:
                continue

        # print(np.array((DBN_data[1])).shape)
        # print(labels)
        return np.array(DBN_data),labels,str(int(cut_length*100))+"%"

def handle_batch(project,cut_length):
    # 获取每个项目版本号
    project_path = "../PROMISE/promise_data/{}".format(project)
    versions = []
    for root,dirnames,filenames in os.walk(project_path):
        for i in filenames:
            versions.append(i[0:-4])
    # 根据项目名和版本号获取最终的数据，并使用sklearn进行训练测试集划分
    for version in versions:
        print(version)
        DBN_data,labels,length = pre_process(project,version,cut_length)
        ros = ROS(random_state=66)

        train_X,test_X,train_Y,test_Y = train_test_split(DBN_data,labels,test_size=0.2,random_state=66)
        train_X_resample,train_Y_resample = ros.fit_resample(train_X,np.array(train_Y))

        # 构建dbn模型进行训练
        # temp_vector=train_X[1]
        max_num = 0
        with open("../vocabdict/{}_{}.pkl".format(project,version),'rb') as f:
            vocabdict = pickle.load(f)
            max_num = len(vocabdict)
        train_X_resample = train_X_resample / (max_num)
        test_X = test_X / max_num
        # train_X=train_X/(np.max(train_X))
        # print(np.linalg.norm(train_X - temp_vector, axis=1))
        # exit()
        within_dbn = UnsupervisedDBN(hidden_layers_structure=[512,256,100],
                                    batch_size=128,
                                    learning_rate_rbm=0.06,
                                    n_epochs_rbm=100,
                                    activation_function='sigmoid',
                                    verbose=False)

        print(train_X_resample.shape)
        within_dbn_input = train_X_resample

        print('{} within DBN train started'.format(project))
        within_dbn.fit(within_dbn_input)

        print('{} within DBN train ended'.format(project))
        within_train_dbn_output = within_dbn.transform(within_dbn_input)
        print(within_train_dbn_output.shape)

        np.save('./train_data/{}_{}_train_X_with_ROS_{}.npy'.format(project,version,length), within_train_dbn_output)

        np.save('./train_data/{}_{}_train_Y_with_ROS_{}.npy'.format(project,version,length), train_Y_resample)

        within_test_dbn_output = within_dbn.transform(test_X)
        np.save('./test_data/{}_{}_test_X_with_ROS_{}.npy'.format(project,version,length), within_test_dbn_output)

        np.save('./test_data/{}_{}_test_Y_with_ROS_{}.npy'.format(project,version,length), np.array(test_Y))



# pre_process("lucene",'2.4')
projects = ['jEdit','ant','ivy']
for i in projects:
    handle_batch(i,1)
# handle_batch("Jedit")