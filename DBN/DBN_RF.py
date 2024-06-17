
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,confusion_matrix,roc_auc_score
import keras
from keras.layers import Input,Embedding,LSTM,Dense,Activation,Multiply, Masking
from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import backend as K
from keras.backend import clear_session
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
import numpy as np
from sklearn import linear_model, manifold
from sklearn.utils import compute_class_weight
from multiprocessing import Process
import time
import math
import os


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
# set_session(sess)
# # set GPU memory
# if('tensorflow' == K.backend()):

#     config = tf.compat.v1.ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = tf.compat.v1.Session(config=config)

def train_and_predict(project,oversample_type,version,max_length):
    # test_projects = ["ambari", "ant", "felix", "jackrabbit", "jenkins", "lucene"]
    # test_projects = ["ambari", "ant", "aptoide", "camel", "cassandra", "egeria", "felix", "jackrabbit", "jenkins",
    #                  "lucene"]
    # test_projects.remove(project)
    # test_projects = ['ant']
    for i in range(3):
        print("{}_{}_{} start".format(project,version, i))
        train_Y = np.load('./train_data/{}_{}_train_Y_{}_{}.npy'.format(project,version,oversample_type,max_length)).astype(np.float64)
        test_Y = np.load('./test_data/{}_{}_test_Y_{}_{}.npy'.format(project,version,oversample_type,max_length)).astype(np.float64)
        for j in range(len(train_Y)):
            if train_Y[j] != 0:
                train_Y[j] = 1
        for j in range(len(test_Y)):
            if test_Y[j] != 0:
                test_Y[j] = 1
        weight = dict(enumerate(compute_class_weight(class_weight='balanced', classes=[0, 1],
                                                        y=train_Y.tolist())))

        train_X = np.load('./train_data/{}_{}_train_X_{}_{}.npy'.format(project,version,oversample_type,max_length)).astype(np.float64)
        test_X = np.load('./test_data/{}_{}_test_X_{}_{}.npy'.format(project,version,oversample_type,max_length)).astype(np.float64)
        # print(train_X)
        # print(train_X)

        # #----------------逻辑回归-----------------
        # clf = linear_model.LogisticRegression(class_weight=weight,max_iter=1000)
        # clf.fit(train_X, train_Y)
        # predict_y_LR = clf.predict_proba(test_X)
        # np.save('./Deeper_LR/{}_{}_{}.npy'.format(project,version,i), predict_y_LR)
        # predict_y_LR=(predict_y_LR[:,1:]-predict_y_LR[:,0:1])/2+0.5
        # predict_y_LR = np.round(predict_y_LR)
        # # print(predict_y_LR.T)
        # confusion_matrix_LR = confusion_matrix(test_Y,predict_y_LR)
        # TP = confusion_matrix_LR[1][1]
        # FP = confusion_matrix_LR[0][1]
        # TN = confusion_matrix_LR[0][0]
        # FN = confusion_matrix_LR[1][0]
        # MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        # with open('./Deeper_LR/res.txt', 'a+', encoding='utf-8') as f:
        #     f.write('{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
        #         project,
        #         version,
        #         str(i),
        #         oversample_type,
        #         max_length,
        #         precision_score(y_true=test_Y, y_pred=predict_y_LR),
        #         recall_score(y_true=test_Y, y_pred=predict_y_LR),
        #         f1_score(y_true=test_Y, y_pred=predict_y_LR),
        #         accuracy_score(y_true=test_Y, y_pred=predict_y_LR),
        #         roc_auc_score(test_Y,predict_y_LR),
        #         MCC
        #     ))

        # #----------------随机森林------------------
        # rfc = RandomForestClassifier(class_weight=weight)
        # rfc.fit(train_X,train_Y)
        # predict_y_RF = rfc.predict_proba(test_X)
        # np.save('./Deeper_RF/{}_{}_{}.npy'.format(project,version, i), predict_y_RF)
        # predict_y_RF = predict_y_RF[:,1:]
        # predict_y_RF = np.round(predict_y_RF)
        # confusion_matrix_RF = confusion_matrix(test_Y,predict_y_RF)
        # TP = confusion_matrix_RF[1][1]
        # FP = confusion_matrix_RF[0][1]
        # TN = confusion_matrix_RF[0][0]
        # FN = confusion_matrix_RF[1][0]
        # MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        # with open('./Deeper_RF/res.txt', 'a+', encoding='utf-8') as f:
        #     f.write('{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
        #         project,
        #         version,
        #         str(i),
        #         oversample_type,
        #         max_length,
        #         precision_score(y_true=test_Y, y_pred=predict_y_RF),
        #         recall_score(y_true=test_Y, y_pred=predict_y_RF),
        #         f1_score(y_true=test_Y, y_pred=predict_y_RF),
        #         accuracy_score(y_true=test_Y, y_pred=predict_y_RF),
        #         roc_auc_score(test_Y, predict_y_RF),
        #         MCC
        #     ))

        #------------------------LSTM------------------------------
        lstm_weight = dict(enumerate(compute_class_weight(class_weight='balanced', classes=[0, 1],
                                                        y=train_Y.tolist())))
        print(lstm_weight)

        # print(train_X.shape)
        traditional_input = Input(shape=(100,1),name='input')
        traditional_lstm_out = LSTM(128,dropout=0.2,recurrent_dropout=0.2,name='promise_lstm')(traditional_input)
        traditional_gate = Dense(128,activation='sigmoid',name='traditional_gate')(traditional_lstm_out)
        traditional_gated_res = Multiply(name='traditional_gated_res')([traditional_gate,traditional_lstm_out])
        main_output = Dense(1,activation='sigmoid',name='main_output')(traditional_gated_res)

        model = Model(inputs=[traditional_input], outputs=[main_output])
        # early_stop = EarlyStopping(monitor='val_f1',patience=15, restore_best_weights=True, mode=max)
        model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy', f1])

        val_data = ({'input': test_X},
                    {'main_output': test_Y})
        model.fit(x={'input': train_X},
                  y={'main_output': train_Y},
                  batch_size=256,
                  epochs=200,
                #   class_weight=lstm_weight,
                  validation_data=val_data)

        predict_y = model.predict(x={'input': test_X})
        np.save('./Deeper_LSTM/{}_{}_{}.npy'.format(project, version, i), predict_y)
        predict_y_LSTM = np.round(predict_y)
        confusion_matrix_LSTM = confusion_matrix(test_Y,predict_y_LSTM)
        TP = confusion_matrix_LSTM[1][1]
        FP = confusion_matrix_LSTM[0][1]
        TN = confusion_matrix_LSTM[0][0]
        FN = confusion_matrix_LSTM[1][0]
        MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        with open('./Deeper_LSTM/res.txt', 'a+', encoding='utf-8') as f:
            f.write('{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
                project,
                version,
                str(i),
                oversample_type,
                max_length,
                precision_score(y_true=test_Y, y_pred=predict_y_LSTM),
                recall_score(y_true=test_Y, y_pred=predict_y_LSTM),
                f1_score(y_true=test_Y, y_pred=predict_y_LSTM),
                accuracy_score(y_true=test_Y, y_pred=predict_y_LSTM),
                roc_auc_score(test_Y, predict_y_LSTM),
                MCC
           ))
    # with open('./Deeper_LR/res.txt', 'a+', encoding='utf-8') as f:
    #     f.write('\n')
    # with open('./Deeper_RF/res.txt', 'a+', encoding='utf-8') as f:
    #     f.write('\n')
    with open('./Deeper_LSTM/res.txt', 'a+', encoding='utf-8') as f:
        f.write('\n')

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

# def mcc(y_true, y_pred):
#     y_pred = K.round(y_pred)
#     tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
#     tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
#     fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
#     fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
#     MCC = (tp*tn-fp*fn)/(K.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+K.epsilon())

#     MCC = tf.where(tf.math.is_nan(MCC), tf.zeros_like(MCC), MCC)
#     return K.mean(MCC)

# train_and_predict("ivy","without_mutation","2.0","100%")
# train_and_predict("ivy","with_mutation","2.0","100%")
# train_and_predict("ivy","with_ROS","2.0","100%")

if __name__ == '__main__':
    projects = ['jEdit','ant','ivy']
    # projects = ['jEdit']
    # Processes = []
    for project in projects:
        project_path = "../PROMISE/promise_data/{}".format(project)
        versions = []
        for root,dirnames,filenames in os.walk(project_path):
            for i in filenames:
                versions.append(i[0:-4])
        # 根据项目名和版本号获取最终的数据，并使用sklearn进行训练测试集划分
        for version in versions:
            # Processes.append(Process(target=train_and_predict,args=(project,"without_mutation",version,"100%")))
            # Processes.append(Process(target=train_and_predict,args=(project,"with_mutation",version,"100%")))
            # Processes.append(Process(target=train_and_predict,args=(project,"with_ROS",version,"100%")))
            # Processes.append(Process(target=train_and_predict,args=(project,"with_manual_mutation",version,"100%")))
            train_and_predict(project,"without_mutation",version,"0.1")
            train_and_predict(project,"with_mutation",version,"0.1")
            train_and_predict(project,"with_ROS",version,"0.1")
            train_and_predict(project,"with_manual_mutation",version,"0.1")
    # for process in Processes:
    #     process.start()
    # for process in Processes:
    #     process.join()