
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,confusion_matrix,roc_auc_score,matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB,BernoulliNB,ComplementNB,CategoricalNB,MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
# import keras
# from keras.layers import Input,Embedding,LSTM,Dense,Activation,Multiply, Masking
# from keras.models import Model
# from keras.callbacks import EarlyStopping,ModelCheckpoint
# from keras import backend as K
# from keras.backend import clear_session
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
import numpy as np
from sklearn import linear_model, manifold
from sklearn.utils import compute_class_weight,shuffle
from multiprocessing import Process
import time
import math
import os
import random


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# config.log_device_placement = True
# sess = tf.compat.v1.Session(config=config)
# set_session(sess)
# # set GPU memory
# if('tensorflow' == K.backend()):

#     config = tf.compat.v1.ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = tf.compat.v1.Session(config=config)

def classify_by_MLP(project,train_version,test_version, oversample_type,max_length,loop_num,train_X,train_Y,test_X,test_Y,suffix):
    # LR_sum_list = [0 for i in range(6)]
    p_list = []
    r_list = []
    f1_list = []
    acc_list = []
    auc_list = []
    mcc_list = []
    for i in range(loop_num):
        print("{}_{}_{}_{}_{}_MLP start".format(project,train_version,test_version, oversample_type, i))

        # X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.01)
        #----------------多层感知机-----------------
        mlp = MLPClassifier(max_iter=1000)
        mlp.fit(train_X, train_Y)
        predict_y_MLP = mlp.predict_proba(test_X)
        # np.save('./result_μbert/Deeper_MLP/{}_{}_{}_{}.npy'.format(project,train_version,test_version,i), predict_y_MLP)

        predict_y_MLP=(predict_y_MLP[:,1:]-predict_y_MLP[:,0:1])/2+0.5
        predict_y_MLP = np.round(predict_y_MLP)
        if matthews_corrcoef(y_true=test_Y, y_pred=predict_y_MLP)<=0.0 and i != loop_num-1:
            continue
        mcc_list.append(matthews_corrcoef(y_true=test_Y, y_pred=predict_y_MLP))
        p_list.append(precision_score(y_true=test_Y, y_pred=predict_y_MLP))
        r_list.append(recall_score(y_true=test_Y, y_pred=predict_y_MLP))
        f1_list.append(f1_score(y_true=test_Y, y_pred=predict_y_MLP))
        acc_list.append(accuracy_score(y_true=test_Y, y_pred=predict_y_MLP))
        auc_list.append(roc_auc_score(test_Y, predict_y_MLP))
    #     LR_sum_list[0] += precision_score(y_true=test_Y, y_pred=predict_y_LR)
    #     LR_sum_list[1] += recall_score(y_true=test_Y, y_pred=predict_y_LR)
    #     LR_sum_list[2] += f1_score(y_true=test_Y, y_pred=predict_y_LR)
    #     LR_sum_list[3] += accuracy_score(y_true=test_Y, y_pred=predict_y_LR)
    #     LR_sum_list[4] += roc_auc_score(test_Y,predict_y_LR)
    #     LR_sum_list[5] += MCC_LR
        with open('./result_μbert/Deeper_MLP/res_by_version_{}_median_value_{}.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
            f.write('{}_{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
                project,
                train_version,test_version,
                str(i),
                oversample_type,
                max_length,
                precision_score(y_true=test_Y, y_pred=predict_y_MLP),
                recall_score(y_true=test_Y, y_pred=predict_y_MLP),
                f1_score(y_true=test_Y, y_pred=predict_y_MLP),
                accuracy_score(y_true=test_Y, y_pred=predict_y_MLP),
                roc_auc_score(test_Y, predict_y_MLP),
                matthews_corrcoef(y_true=test_Y, y_pred=predict_y_MLP)
            ))
    with open('./result_μbert/Deeper_MLP/res_by_version_{}_median_value_{}.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
        f.write('\n')
    # LR_avg = np.array(LR_sum_list)
    with open('./result_μbert/Deeper_MLP/res_by_version_{}_{}.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
        # f.write('\n')
        f.write('{}_{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
            project,
            train_version,test_version,
            "avg",
            oversample_type,
            max_length,
            np.array(p_list).mean(),
            np.array(r_list).mean(),
            np.array(f1_list).mean(),
            np.array(acc_list).mean(),
            np.array(auc_list).mean(),
            np.array(mcc_list).mean()
        ))
        f.write('\n')



def classify_by_LR(project,train_version,test_version, oversample_type,max_length,loop_num,train_X,train_Y,test_X,test_Y,suffix):
    # LR_sum_list = [0 for i in range(6)]
    p_list = []
    r_list = []
    f1_list = []
    acc_list = []
    auc_list = []
    mcc_list = []
    for i in range(loop_num):
        print("{}_{}_{}_{}_{}_LR start".format(project,train_version,test_version, oversample_type, i))

        # X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.01)
        #----------------逻辑回归-----------------
        clf = linear_model.LogisticRegression(max_iter=1000)
        # train_X,train_Y = shuffle(train_X, train_Y)
        clf.fit(train_X, train_Y)
        # clf.fit(X_train, y_train)
        predict_y_LR = clf.predict_proba(test_X)
        # np.save('./result_μbert/Deeper_LR/{}_{}_{}_{}.npy'.format(project,train_version,test_version,i), predict_y_LR)
        # print(predict_y_LR)
        predict_y_LR=(predict_y_LR[:,1:]-predict_y_LR[:,0:1])/2+0.5
        predict_y_LR = np.round(predict_y_LR)
        # print(predict_y_LR.T)
        confusion_matrix_LR = confusion_matrix(test_Y,predict_y_LR)
        TP = confusion_matrix_LR[1][1]
        FP = confusion_matrix_LR[0][1]
        TN = confusion_matrix_LR[0][0]
        FN = confusion_matrix_LR[1][0]
        MCC_LR = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        mcc_list.append(MCC_LR)
        p_list.append(precision_score(y_true=test_Y, y_pred=predict_y_LR))
        r_list.append(recall_score(y_true=test_Y, y_pred=predict_y_LR))
        f1_list.append(f1_score(y_true=test_Y, y_pred=predict_y_LR))
        acc_list.append(accuracy_score(y_true=test_Y, y_pred=predict_y_LR))
        auc_list.append(roc_auc_score(test_Y, predict_y_LR))
    #     LR_sum_list[0] += precision_score(y_true=test_Y, y_pred=predict_y_LR)
    #     LR_sum_list[1] += recall_score(y_true=test_Y, y_pred=predict_y_LR)
    #     LR_sum_list[2] += f1_score(y_true=test_Y, y_pred=predict_y_LR)
    #     LR_sum_list[3] += accuracy_score(y_true=test_Y, y_pred=predict_y_LR)
    #     LR_sum_list[4] += roc_auc_score(test_Y,predict_y_LR)
    #     LR_sum_list[5] += MCC_LR
        with open('./result_μbert/Deeper_LR/res_by_version_{}_median_value_{}.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
            f.write('{}_{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
                project,
                train_version,test_version,
                str(i),
                oversample_type,
                max_length,
                precision_score(y_true=test_Y, y_pred=predict_y_LR),
                recall_score(y_true=test_Y, y_pred=predict_y_LR),
                f1_score(y_true=test_Y, y_pred=predict_y_LR),
                accuracy_score(y_true=test_Y, y_pred=predict_y_LR),
                roc_auc_score(test_Y, predict_y_LR),
                MCC_LR
            ))
    with open('./result_μbert/Deeper_LR/res_by_version_{}_median_value_{}.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
        f.write('\n')
    # LR_avg = np.array(LR_sum_list)
    with open('./result_μbert/Deeper_LR/res_by_version_{}_{}.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
        # f.write('\n')
        f.write('{}_{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
            project,
            train_version,test_version,
            "avg",
            oversample_type,
            max_length,
            np.array(p_list).mean(),
            np.array(r_list).mean(),
            np.array(f1_list).mean(),
            np.array(acc_list).mean(),
            np.array(auc_list).mean(),
            np.array(mcc_list).mean()
        ))
        f.write('\n')

def classify_by_RF(project,train_version,test_version, oversample_type,max_length,loop_num,train_X,train_Y,test_X,test_Y,suffix):
    # RF_sum_list = [0 for i in range(6)]
    p_list = []
    r_list = []
    f1_list = []
    acc_list = []
    auc_list = []
    mcc_list = []
    for i in range(loop_num):
        print("{}_{}_{}_{}_{}_RF start".format(project,train_version,test_version, oversample_type, i))


        #----------------随机森林------------------
        rfc = RandomForestClassifier()
        rfc.fit(train_X,train_Y)
        predict_y_RF = rfc.predict_proba(test_X)
        # np.save('./result_μbert/Deeper_RF/{}_{}_{}_{}.npy'.format(project,train_version,test_version, i), predict_y_RF)
        # print(predict_y_RF)
        predict_y_RF = predict_y_RF[:,1:]
        predict_y_RF = np.round(predict_y_RF)
        confusion_matrix_RF = confusion_matrix(test_Y,predict_y_RF)
        TP = confusion_matrix_RF[1][1]
        FP = confusion_matrix_RF[0][1]
        TN = confusion_matrix_RF[0][0]
        FN = confusion_matrix_RF[1][0]
        MCC_RF = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        mcc_list.append(MCC_RF)
        p_list.append(precision_score(y_true=test_Y, y_pred=predict_y_RF))
        r_list.append(recall_score(y_true=test_Y, y_pred=predict_y_RF))
        f1_list.append(f1_score(y_true=test_Y, y_pred=predict_y_RF))
        acc_list.append(accuracy_score(y_true=test_Y, y_pred=predict_y_RF))
        auc_list.append(roc_auc_score(test_Y, predict_y_RF))
        # RF_sum_list[0] += precision_score(y_true=test_Y, y_pred=predict_y_RF)
        # RF_sum_list[1] += recall_score(y_true=test_Y, y_pred=predict_y_RF)
        # RF_sum_list[2] += f1_score(y_true=test_Y, y_pred=predict_y_RF)
        # RF_sum_list[3] += accuracy_score(y_true=test_Y, y_pred=predict_y_RF)
        # RF_sum_list[4] += roc_auc_score(test_Y,predict_y_RF)
        # RF_sum_list[5] += MCC_RF

        with open('./result_μbert/Deeper_RF/res_by_version_{}_median_value_{}_100.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
            f.write('{}_{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
                project,
                train_version,test_version,
                str(i),
                oversample_type,
                max_length,
                precision_score(y_true=test_Y, y_pred=predict_y_RF),
                recall_score(y_true=test_Y, y_pred=predict_y_RF),
                f1_score(y_true=test_Y, y_pred=predict_y_RF),
                accuracy_score(y_true=test_Y, y_pred=predict_y_RF),
                roc_auc_score(test_Y, predict_y_RF),
                MCC_RF
            ))
    with open('./result_μbert/Deeper_RF/res_by_version_{}_median_value_{}_100.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
        f.write('\n')
    # RF_avg = np.array(RF_sum_list)/loop_num
    with open('./result_μbert/Deeper_RF/res_by_version_{}_{}_100.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
        # f.write('\n')
        f.write('{}_{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
            project,
            train_version,test_version,
            "avg",
            oversample_type,
            max_length,
            np.array(p_list).mean(),
            np.array(r_list).mean(),
            np.array(f1_list).mean(),
            np.array(acc_list).mean(),
            np.array(auc_list).mean(),
            np.array(mcc_list).mean()
        ))
        f.write('\n')

def classify_by_KNN(project,train_version,test_version, oversample_type,max_length,loop_num,train_X,train_Y,test_X,test_Y,suffix):
    # KNN_sum_list = [0 for i in range(6)]
    p_list = []
    r_list = []
    f1_list = []
    acc_list = []
    auc_list = []
    mcc_list = []
    for i in range(loop_num):
        print("{}_{}_{}_{}_{}_KNN start".format(project,train_version,test_version, oversample_type, i))

        # X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.01)
        #----------------KNN-----------------   
        knn = KNeighborsClassifier()
        knn.fit(train_X, train_Y)
        # knn.fit(X_train, y_train)
        predict_y_KNN = knn.predict(test_X)
        # np.save('./result_μbert/Deeper_KNN/{}_{}_{}_{}.npy'.format(project,train_version,test_version,i), predict_y_KNN)
        # print(type(predict_y_KNN))
        # predict_y_KNN=(predict_y_KNN[:,1:]-predict_y_KNN[:,0:1])/2+0.5
        # predict_y_KNN = np.round(predict_y_KNN)
        # print(type(predict_y_KNN))
        # print(predict_y_LR.T)
        confusion_matrix_KNN = confusion_matrix(test_Y,predict_y_KNN)
        TP = confusion_matrix_KNN[1][1]
        FP = confusion_matrix_KNN[0][1]
        TN = confusion_matrix_KNN[0][0]
        FN = confusion_matrix_KNN[1][0]
        MCC_KNN = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        mcc_list.append(MCC_KNN)
        p_list.append(precision_score(y_true=test_Y, y_pred=predict_y_KNN))
        r_list.append(recall_score(y_true=test_Y, y_pred=predict_y_KNN))
        f1_list.append(f1_score(y_true=test_Y, y_pred=predict_y_KNN))
        acc_list.append(accuracy_score(y_true=test_Y, y_pred=predict_y_KNN))
        auc_list.append(roc_auc_score(test_Y, predict_y_KNN))
    #     KNN_sum_list[0] += precision_score(y_true=test_Y, y_pred=predict_y_KNN)
    #     KNN_sum_list[1] += recall_score(y_true=test_Y, y_pred=predict_y_KNN)
    #     KNN_sum_list[2] += f1_score(y_true=test_Y, y_pred=predict_y_KNN)
    #     KNN_sum_list[3] += accuracy_score(y_true=test_Y, y_pred=predict_y_KNN)
    #     KNN_sum_list[4] += roc_auc_score(test_Y,predict_y_KNN)
    #     KNN_sum_list[5] += MCC_KNN
        with open('./result_μbert/Deeper_KNN/res_by_version_{}_median_value_{}.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
            # f.write('\n')
            f.write('{}_{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
                project,
                train_version,test_version,
                str(i),
                oversample_type,
                max_length,
                precision_score(y_true=test_Y, y_pred=predict_y_KNN),
                recall_score(y_true=test_Y, y_pred=predict_y_KNN),
                f1_score(y_true=test_Y, y_pred=predict_y_KNN),
                accuracy_score(y_true=test_Y, y_pred=predict_y_KNN),
                roc_auc_score(test_Y, predict_y_KNN),
                MCC_KNN
            ))
    with open('./result_μbert/Deeper_KNN/res_by_version_{}_median_value_{}.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
        f.write('\n')
    # KNN_avg = np.array(KNN_sum_list)/loop_num
    with open('./result_μbert/Deeper_KNN/res_by_version_{}_{}.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
        # f.write('\n')
        f.write('{}_{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
            project,
            train_version,test_version,
            "avg",
            oversample_type,
            max_length,
            np.array(p_list).mean(),
            np.array(r_list).mean(),
            np.array(f1_list).mean(),
            np.array(acc_list).mean(),
            np.array(auc_list).mean(),
            np.array(mcc_list).mean()
        ))
        f.write('\n')

def classify_by_SVM(project,train_version,test_version, oversample_type,max_length,loop_num,train_X,train_Y,test_X,test_Y,suffix):
    # SVM_sum_list = [0 for i in range(6)]
    p_list = [] 
    r_list = []
    f1_list = []
    acc_list = []
    auc_list = []
    mcc_list = []
    for i in range(loop_num):
        print("{}_{}_{}_{}_{}_SVM start".format(project,train_version,test_version, oversample_type, i))

        # X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.01)
        #----------------SVM-----------------
        svm = LinearSVC(max_iter=10000)
        # train_X,train_Y = shuffle(train_X, train_Y)
        # print(train_X)
        svm.fit(train_X, train_Y)
        # svm.fit(X_train, y_train)
        predict_y_SVM = svm.predict(test_X)
        # np.save('./result_μbert/Deeper_SVM/{}_{}_{}_{}.npy'.format(project,train_version,test_version,i), predict_y_SVM)
        # print(predict_y_SVM)
        # predict_y_SVM=(predict_y_SVM[:,1:]-predict_y_SVM[:,0:1])/2+0.5
        # predict_y_SVM = np.round(predict_y_SVM)
        # print(predict_y_LR.T)
        confusion_matrix_SVM = confusion_matrix(test_Y,predict_y_SVM)
        TP = confusion_matrix_SVM[1][1]
        FP = confusion_matrix_SVM[0][1]
        TN = confusion_matrix_SVM[0][0]
        FN = confusion_matrix_SVM[1][0]
        MCC_SVM = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        mcc_list.append(MCC_SVM)
        p_list.append(precision_score(y_true=test_Y, y_pred=predict_y_SVM))
        r_list.append(recall_score(y_true=test_Y, y_pred=predict_y_SVM))
        f1_list.append(f1_score(y_true=test_Y, y_pred=predict_y_SVM))
        acc_list.append(accuracy_score(y_true=test_Y, y_pred=predict_y_SVM))
        auc_list.append(roc_auc_score(test_Y, predict_y_SVM))
    #     SVM_sum_list[0] += precision_score(y_true=test_Y, y_pred=predict_y_SVM)
    #     SVM_sum_list[1] += recall_score(y_true=test_Y, y_pred=predict_y_SVM)
    #     SVM_sum_list[2] += f1_score(y_true=test_Y, y_pred=predict_y_SVM)
    #     SVM_sum_list[3] += accuracy_score(y_true=test_Y, y_pred=predict_y_SVM)
    #     SVM_sum_list[4] += roc_auc_score(test_Y,predict_y_SVM)
    #     SVM_sum_list[5] += MCC_SVM
        with open('./result_μbert/Deeper_SVM/res_by_version_{}_median_value_{}.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
            # f.write('\n')
            f.write('{}_{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
                project,
                train_version,test_version,
                str(i),
                oversample_type,
                max_length,
                precision_score(y_true=test_Y, y_pred=predict_y_SVM),
                recall_score(y_true=test_Y, y_pred=predict_y_SVM),
                f1_score(y_true=test_Y, y_pred=predict_y_SVM),
                accuracy_score(y_true=test_Y, y_pred=predict_y_SVM),
                roc_auc_score(test_Y, predict_y_SVM),
                MCC_SVM
            ))
    with open('./result_μbert/Deeper_SVM/res_by_version_{}_median_value_{}.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
        f.write('\n')
    # SVM_avg = np.array(SVM_sum_list)/loop_num
    with open('./result_μbert/Deeper_SVM/res_by_version_{}_{}.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
        # f.write('\n')
        f.write('{}_{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
            project,
            train_version,test_version,
            "avg",
            oversample_type,
            max_length,
            np.array(p_list).mean(),
            np.array(r_list).mean(),
            np.array(f1_list).mean(),
            np.array(acc_list).mean(),
            np.array(auc_list).mean(),
            np.array(mcc_list).mean()
        ))
        f.write('\n')

def classify_by_NB(project,train_version,test_version, oversample_type,max_length,loop_num,train_X,train_Y,test_X,test_Y,suffix):
    # kind_list = [GaussianNB,BernoulliNB,ComplementNB,CategoricalNB,MultinomialNB]
    kind_list = [ComplementNB]
    # kind_rank = {"GaussianNB":0,"BernoulliNB":0,"ComplementNB":0,"CategoricalNB":0,"MultinomialNB":0}
    for j in kind_list:
        # NB_sum_list = [0 for i in range(6)]
        p_list = []
        r_list = []
        f1_list = []
        acc_list = []
        auc_list = []
        mcc_list = []
        mcc_list2 = []
        for i in range(loop_num):
            print("{}_{}_{}_{}_{}_{} start".format(project,train_version,test_version, oversample_type, i, j.__name__))

            # X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.01)
            #----------------NB-----------------
            nb = j()
            # train_X,train_Y = shuffle(train_X, train_Y)
            nb.fit(train_X, train_Y)
            # nb.fit(X_train, y_train)
            predict_y_NB = nb.predict(test_X)
            # np.save('./result_μbert/Deeper_NB/{}_{}_{}_{}_{}.npy'.format(project,train_version,test_version,i,j.__name__), predict_y_NB)
            # predict_y_NB=(predict_y_NB[:,1:]-predict_y_NB[:,0:1])/2+0.5
            # predict_y_NB = np.round(predict_y_NB)
            # print(predict_y_LR.T)
            confusion_matrix_NB = confusion_matrix(test_Y,predict_y_NB)
            TP = confusion_matrix_NB[1][1]
            FP = confusion_matrix_NB[0][1]
            TN = confusion_matrix_NB[0][0]
            FN = confusion_matrix_NB[1][0]
            MCC_NB = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
            mcc_list.append(MCC_NB)
            p_list.append(precision_score(y_true=test_Y, y_pred=predict_y_NB))
            r_list.append(recall_score(y_true=test_Y, y_pred=predict_y_NB))
            f1_list.append(f1_score(y_true=test_Y, y_pred=predict_y_NB))
            acc_list.append(accuracy_score(y_true=test_Y, y_pred=predict_y_NB))
            auc_list.append(roc_auc_score(test_Y, predict_y_NB))
            mcc_list2.append(matthews_corrcoef(test_Y, predict_y_NB))
            # NB_sum_list[0] += precision_score(y_true=test_Y, y_pred=predict_y_NB)
            # NB_sum_list[1] += recall_score(y_true=test_Y, y_pred=predict_y_NB)
            # NB_sum_list[2] += f1_score(y_true=test_Y, y_pred=predict_y_NB)
            # NB_sum_list[3] += accuracy_score(y_true=test_Y, y_pred=predict_y_NB)
            # NB_sum_list[4] += roc_auc_score(test_Y,predict_y_NB)
            # NB_sum_list[5] += MCC_NB
            with open('./result_μbert/Deeper_NB/res_by_version_{}_median_value_{}.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
            # f.write('\n')
                f.write('{}_{}_{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
                    project,
                    train_version,test_version,
                    str(i),
                    oversample_type,
                    max_length,
                    j.__name__,
                    precision_score(y_true=test_Y, y_pred=predict_y_NB),
                    recall_score(y_true=test_Y, y_pred=predict_y_NB),
                    f1_score(y_true=test_Y, y_pred=predict_y_NB),
                    accuracy_score(y_true=test_Y, y_pred=predict_y_NB),
                    roc_auc_score(test_Y, predict_y_NB),
                    matthews_corrcoef(test_Y, predict_y_NB),
                ))
        with open('./result_μbert/Deeper_NB/res_by_version_{}_median_value_{}.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
            f.write('\n')

        with open('./result_μbert/Deeper_NB/res_by_version_{}_{}.txt'.format(max_length,suffix), 'a+', encoding='utf-8') as f:
            # f.write('\n')
            f.write('{}_{}_{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}_{}\n'.format(
                project,
                train_version,test_version,
                "avg",
                oversample_type,
                max_length,
                j.__name__,
                np.array(p_list).mean(),
                np.array(r_list).mean(),
                np.array(f1_list).mean(),
                np.array(acc_list).mean(),
                np.array(auc_list).mean(),
                np.array(mcc_list).mean(),
                np.array(mcc_list2).mean()
            ))
            f.write('\n')

def classify_by_LSTM(project,train_version,test_version, oversample_type,max_length,loop_num,train_X,train_Y,test_X,test_Y):
    p_list = []
    r_list = []
    f1_list = []
    acc_list = []
    auc_list = []
    mcc_list = []
    data_type = "with_DBN"
    for i in range(loop_num):
        traditional_input = Input(shape=(100,1),name='input')
        traditional_lstm_out = LSTM(64,name='promise_lstm')(traditional_input)
        traditional_gate = Dense(64,activation='sigmoid',name='traditional_gate')(traditional_lstm_out)
        traditional_gated_res = Multiply(name='traditional_gated_res')([traditional_gate,traditional_lstm_out])
        # print(traditional_gated_res)
        main_output = Dense(1,activation='sigmoid',name='main_output')(traditional_gated_res)


        model = Model(inputs=[traditional_input], outputs=[main_output])
        early_stopping = EarlyStopping(monitor='val_f1', patience=15, verbose=1)
        model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy',f1])

        val_data = ({'input': test_X},
                    {'main_output': test_Y})
        model.fit(x={'input': train_X},
                    y={'main_output': train_Y},
                    batch_size=64,
                    epochs=100,
                #   class_weight=lstm_weight,
                    # validation_split=0.2,
                    # callbacks=[early_stopping]
                    )

        predict_y = model.predict(x={'input': test_X})
        np.save('./Deeper_LSTM/{}_{}_{}_{}.npy'.format(project, train_version,test_version, data_type), predict_y)
        predict_y_LSTM = np.round(predict_y)
        confusion_matrix_LSTM = confusion_matrix(test_Y,predict_y_LSTM)
        TP = confusion_matrix_LSTM[1][1]
        FP = confusion_matrix_LSTM[0][1]
        TN = confusion_matrix_LSTM[0][0]
        FN = confusion_matrix_LSTM[1][0]
        MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        mcc_list.append(MCC)
        p_list.append(precision_score(y_true=test_Y, y_pred=predict_y_LSTM))
        r_list.append(recall_score(y_true=test_Y, y_pred=predict_y_LSTM))
        f1_list.append(f1_score(y_true=test_Y, y_pred=predict_y_LSTM))
        acc_list.append(accuracy_score(y_true=test_Y, y_pred=predict_y_LSTM))
        auc_list.append(roc_auc_score(test_Y, predict_y_LSTM))
    print(np.array(f1_list).mean())
    print(np.array(auc_list).mean())
    with open('./Deeper_LSTM/res_with_glove_by_version_{}_with_DBN.txt'.format(max_length), 'a+', encoding='utf-8') as f:
        f.write('{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
            project,
            train_version,test_version,
            data_type,
            max_length,
            np.array(p_list).mean(),
            np.array(r_list).mean(),
            np.array(f1_list).mean(),
            np.array(acc_list).mean(),
            np.array(auc_list).mean(),
            np.array(mcc_list).mean()
        ))
        f.write("\n")

def classify_by_GB_SMOTE(project,train_version,test_version, oversample_type,max_length,loop_num,train_X,train_Y,test_X,test_Y,suffix,classify_model):
    if classify_model == "NB":
        model = ComplementNB()
    elif classify_model == "SVM":
        model = LinearSVC()
    elif classify_model == "MLP":
        model = MLPClassifier()
    elif classify_model == "LR":
        model = linear_model.LogisticRegression()
    




def train_and_predict(project,oversample_type,train_version,test_version,max_length,ros_ratio=0,smote_ratio=0,mutation_ratio=0):
    # test_projects = ["ambari", "ant", "felix", "jackrabbit", "jenkins", "lucene"]
    # test_projects = ["ambari", "ant", "aptoide", "camel", "cassandra", "egeria", "felix", "jackrabbit", "jenkins",
    #                  "lucene"]
    # test_projects.remove(project)
    # test_projects = ['ant']
    if oversample_type == "with_mix":
        train_Y = np.load('./train_data_by_version/{}_{}_{}_train_Y_{}_{}_{}_{}_{}.npy'.format(project,train_version,test_version,oversample_type,max_length,ros_ratio,smote_ratio,mutation_ratio)).astype(np.float64)
        train_X = np.load('./train_data_by_version/{}_{}_{}_train_X_{}_{}_{}_{}_{}.npy'.format(project,train_version,test_version,oversample_type,max_length,ros_ratio,smote_ratio,mutation_ratio)).astype(np.float64)
    else:
        train_Y = np.load('./train_data_by_version/{}_{}_{}_train_Y_{}_{}.npy'.format(project,train_version,test_version,oversample_type,max_length)).astype(np.float64)
        train_X = np.load('./train_data_by_version/{}_{}_{}_train_X_{}_{}.npy'.format(project,train_version,test_version,oversample_type,max_length)).astype(np.float64)
    test_Y = np.load('./test_data_by_version/{}_{}_{}_test_Y_{}_{}.npy'.format(project,train_version,test_version,oversample_type,max_length)).astype(np.float64)

    # if(oversample_type == "without_mutation"):
    #     with open('./train_defect_ratio.txt','a+') as f:
    #         f.write("{}_{}:  ".format(project,train_version)+str(np.sum(train_Y==1)/len(train_Y)))
    #         f.write("\n")

    # weight = dict(enumerate(compute_class_weight(class_weight='balanced', classes=[0, 1],
    #                                                 y=train_Y.tolist())))

    test_X = np.load('./test_data_by_version/{}_{}_{}_test_X_{}_{}.npy'.format(project,train_version,test_version,oversample_type,max_length)).astype(np.float64)
    # print(train_X)
    # # print(train_X)
    # suffix = "graphc_graphc_with_ast_length128_0.9_1"
    # suffix = "graphc_codebert-mlm_with_ast_length128_0.9_1"
    # suffix = "graphc_codebert-java_with_ast_length128_0.9"
    # suffix = "graphc_length_32_0.9"
    # suffix = "graphc_length_64_0.9"
    # suffix = "graphc_length_128_0.9"
    suffix = "ASNSMOTE"
    classify_by_LR(project,train_version,test_version,oversample_type,max_length,30,train_X,train_Y,test_X,test_Y,suffix)
    # classify_by_RF(project,train_version,test_version,oversample_type,max_length,50,train_X,train_Y,test_X,test_Y,suffix)
    # classify_by_KNN(project,train_version,test_version,oversample_type,max_length,30,train_X,train_Y,test_X,test_Y,suffix)
    classify_by_SVM(project,train_version,test_version,oversample_type,max_length,30,train_X,train_Y,test_X,test_Y,suffix)
    classify_by_NB(project,train_version,test_version,oversample_type,max_length,30,train_X,train_Y,test_X,test_Y,suffix)
    classify_by_MLP(project,train_version,test_version,oversample_type,max_length,50,train_X,train_Y,test_X,test_Y,suffix)
    # classify_by_LSTM(project,train_version,test_version,oversample_type,max_length,5,train_X,train_Y,test_X,test_Y)


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


if __name__ == '__main__':
    # # projects = ['jEdit','ant','ivy']
    # projects_with_version = {'ant':['1.6','1.7'],"jEdit":['4.2','4.3'],"synapse":['1.0','1.1','1.2'],"camel":['1.4','1.6'],"ivy":['1.4','2.0']}
    # projects_with_version = {'ant':['1.5','1.6','1.7'],"jEdit":['4.0','4.1','4.2','4.3'],"synapse":['1.0','1.1','1.2'],"camel":['1.4','1.6'],
    #                         "ivy":['1.4','2.0'],"poi":['2.0','2.5'],"xalan":['2.4','2.5'],"xerces":['1.2','1.3'],"log4j":['1.0','1.1']}
    # projects_with_version = {"synapse":['1.1','1.2'],"xalan":['2.4','2.5']}
    # projects_with_version = {'ant':['1.5','1.6','1.7'],"jEdit":['4.0','4.1','4.2'],"synapse":['1.1','1.2'],"xalan":['2.4','2.5']}
    projects_with_version = {'ant':['1.5','1.6','1.7'],"jEdit":['4.0','4.1','4.2'],"synapse":['1.0','1.1','1.2'],"camel":['1.4','1.6'],
                                "ivy":['1.4','2.0'],"xalan":['2.4','2.5']}
    # projects_with_version = {"ant":['1.5','1.6'],"jEdit":['4.1','4.2'],"camel":['1.4','1.6'],"xalan":['2.4','2.5']}
    # projects_with_version = {'ivy':['1.4','2.0']}
    Processes = []
    for project in projects_with_version.keys():
        project_path = "../PROMISE/promise_data/{}".format(project)
        versions = projects_with_version[project]
        # for root,dirnames,filenames in os.walk(project_path):
        #     for i in filenames:
        #         versions.append(i[0:-4])
        # 根据项目名和版本号获取最终的数据，并使用sklearn进行训练测试集划分
        for i in range(len(versions)-1):
            train_version = versions[i]
            test_version = versions[i+1]
            # Processes.append(Process(target=train_and_predict,args=(project,"without_mutation",train_version, test_version,"50%")))
            # # Processes.append(Process(target=train_and_predict,args=(project,"with_mutation",train_version, test_version,"50%")))
            # Processes.append(Process(target=train_and_predict,args=(project,"with_ROS",train_version, test_version,"50%")))
            # Processes.append(Process(target=train_and_predict,args=(project,"with_SMOTE",train_version, test_version,"50%")))
            # Processes.append(Process(target=train_and_predict,args=(project,"with_manual_mutation",train_version, test_version,"50%")))
            # train_and_predict(project,"without_mutation",train_version, test_version,"50%")
            # train_and_predict(project,"with_mutation",train_version, test_version,"0.1")1
            # train_and_predict(project,"with_ROS",train_version, test_version,"50%")
            # train_and_predict(project,"with_SMOTE",train_version, test_version,"50%")
            # train_and_predict(project,"with_KMeansSMOTE",train_version, test_version,"50%")
            train_and_predict(project,"with_ASNSMOTE",train_version, test_version,"50%")
            # train_and_predict(project,"with_manual_mutation",train_version, test_version,"80%")
            # oversample_type = f"with_μbert_mutation_graphc_graphc_with_ast_length125_0.9"
            # oversample_type = f"with_μbert_mutation_graphc_codebert-mlm_with_ast_length125_0.9"
            # oversample_type = f"with_μbert_mutation_graphc_codebert-java_with_ast_length125_0.9"
            # oversample_type = f"with_μbert_mutation_graphc_length_32_0.9"
            # oversample_type = f"with_μbert_mutation_graphc_length_64_0.9"
            # oversample_type = f"with_μbert_mutation_graphc_length_128_0.9"
            # oversample_type = f"with_μbert_mutation_graphc_not_use_0.95"
            # train_and_predict(project,oversample_type,train_version, test_version,"20%")
            # train_and_predict(project,"with_μbert_mutation_graphc_use_0.95",train_version, test_version,"50%") #使用graphcodebert+ast序列
            # train_and_predict(project,"with_μbert_mutation_graphc_0.95",train_version, test_version,"50%") #使用graphcodebert不加ast序列
            # train_and_predict(project,"with_DBN",train_version, test_version,"100%")
            # train_and_predict(project,"with_mix",train_version, test_version,"100%",0,0.25,0.75)
    # for process in Processes:
    #     process.start()
    # for process in Processes:
    #     process.join()