#标准化工具
from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#导入集合分割，交叉验证，网格搜索
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,cross_validate,KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score,roc_curve,auc,confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#smote过采样
# from imblearn.over_sampling import SMOTE,BorderlineSMOTE,SVMSMOTE,ADASYN,KMeansSMOTE,RandomOverSampler
# #欠采样
# from imblearn.under_sampling import RandomUnderSampler
import random
import math
import heapq
import time
def Euclidean_Metric(a,b):
      """
      欧式距离
      """
      dis=0
      A = np.array(a)
      B = np.array(b)
      n=A.shape[0]
      for i in range(n):
          dis=dis+(A[i]-B[i])*(A[i]-B[i])
      dis=np.sqrt(dis)
      return dis
def generate_x(N,k,train_x,train_y):
    #n=int(N/10)
    time_start=time.time()
    g_index=0
    wrg=0
    samples_X=train_x
    samples_Y=train_y
    Minority_sample_X = []
    for index,msg in enumerate(samples_Y):
        if(msg==1):
            Minority_sample_X.append(samples_X[index])
    # Minority_sample=samples[samples.iloc[:,-1].isin(['1'])]
    # Minority_sample_X=Minority_sample.iloc[:,0:-1]
                                       
    # transfer = StandardScaler()
    # SMinority_X= transfer.fit_transform(Minority_sample)
    # All_X=transfer.fit_transform(samples_X)
    Minority_X=np.array(Minority_sample_X)
    All_X=np.array(samples_X)
    n1=All_X.shape[0]-2*Minority_X.shape[0]
    # print(n1)
    #n=int((All_X.shape[0]-2*Minority_X.shape[0])/Minority_X.shape[0])
    #print(n)
    dis_matrix=np.zeros((Minority_X.shape[0],All_X.shape[0]),dtype=float)
    for i in range(0,Minority_X.shape[0]):
        for j in range(0,All_X.shape[0]):
            dis_matrix[i,j]=Euclidean_Metric(Minority_X[i,:],All_X[j,:])
            if(dis_matrix[i,j]==0):
                dis_matrix[i,j]=999999
    dis_matrix=dis_matrix.tolist()
    
    d=[]
    #print(Minority_X.shape[0])
    for i in range(Minority_X.shape[0]):
        min_index=list(map(dis_matrix[i].index, heapq.nsmallest(1, dis_matrix[i])))
        # print(min_index)
        if(samples_Y[min_index[0]]==0): 
            d.append(i)
    # print(Minority_X)
    Minority_X=np.delete(Minority_X,d,axis=0)
    # print(Minority_X)
    #print(Minority_X.shape)
    n=int((n1)/Minority_X.shape[0])
    #print(n)
    synthetic = np.zeros(((Minority_X.shape[0])*n,Minority_X.shape[1]),dtype=float)
    #print(Minority_X.shape[0])
    for i in range(Minority_X.shape[0]):
        min_index=list(map(dis_matrix[i].index, heapq.nsmallest(k, dis_matrix[i])))
        best_index={}
        best_f=0
        for h in range(len(min_index)):
            
            if(samples_Y[min_index[h]]==0):
               best_index[best_f]=min_index[h]
               best_f+=1
               break
            else:
                best_index[best_f]=min_index[h]
                best_f+=1
        #print(best_index)
        for j in range(0,n):
            nn=random.randint(0,len(best_index)-1)
            #print(min_index[nn])
            dif=All_X[best_index[nn]]-Minority_X[i]
            #print(dif)
            gap=random.random()
            synthetic[g_index]=Minority_X[i]+gap*dif
            g_index+=1
            
    #print(synthetic.shape)
    #print(wrg)
    
    # synthetic=synthetic[0:synthetic.shape[0]-,:]
    new_train_x = np.array(train_x)
    new_train_y = np.array(train_y)
    labels=np.ones(synthetic.shape[0])
    new_train_y = np.concatenate((new_train_y,labels),axis=0)
    # synthetic=np.insert(synthetic,synthetic.shape[1],values=labels,axis=1)
    new_train_x=np.concatenate((new_train_x,synthetic),axis=0)
    time_end=time.time()
    del(dis_matrix)
    return new_train_x,new_train_y
