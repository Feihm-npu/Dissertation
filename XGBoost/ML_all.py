### Import important packages ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold,ShuffleSplit
from sklearn.metrics import auc, plot_roc_curve, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB,CategoricalNB
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
from arbiter_PUF import*
from XOR_PUF import*
from lightweight_PUF import*
from feedforward_PUF import*
from interpose_PUF import*
from LFSR_simulated import*
from Puf_resilience import*
from general_model import*
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

'''puf_seeds = [11, 67, 82, 95, 324, 56, 70, 12, 123, 39]
train_challenge_seeds = [123, 21, 34, 54, 221, 345, 676, 54, 33, 15]
test_challenge_seeds = [19, 44, 1, 77, 453, 657, 557, 23, 44, 84]
clf_result = pd.DataFrame({'puf_seed' : [],
                           'train_challenge_seed': [],
                           'test_challenge_seed': [],
                           'Test split Accuracy' : [],
                           'Test split F1' : [],
                           'Cross_val Accuracy' : [],
                           'Cross_val sd' : [],
                           'Cross_val split F1' : [],
                           'Cross_val F1 sd' : [],
                           'Training time' : [],
                           'Testing time' : []
                           })'''

PUF_types = [    #'APUF' #'APUF','2 XOR-APUF','3 XOR-APUF','4 XOR-APUF','5 XOR-APUF','6 XOR-APUF','FF-2-XOR-APUF','FF-3-XOR-APUF',
                 #'FF-4-XOR-APUF','FF-5-XOR-APUF','FF-6-XOR-APUF',
                 'Generic framework(1,1,1)'#'Generic framework(2,2,2)','Generic framework(3,3,3)',
                 #'Generic framework(4,4,4)','Generic framework(5,5,5)','Generic framework(6,6,6)','Generic framework(10,10,10)'
                 #,'Generic framework(20,20,20)'
                 #'Generic framework(2,0,0)','Generic framework(3,0,0)','Generic framework(4,0,0)','Generic framework(5,0,0)',
                 #'Generic framework(6,0,0)','Generic framework(7,0,0)','Generic framework(8,0,0)','Generic framework(9,0,0)',
                 #'Generic framework(10,0,0)','Generic framework(20,0,0)',
                 #'Generic framework(0,2,0)','Generic framework(0,3,0)','Generic framework(0,4,0)','Generic framework(0,5,0)',
                # 'Generic framework(0,6,0)','Generic framework(0,7,0)','Generic framework(0,8,0)','Generic framework(0,9,0)',
                # 'Generic framework(0,10,0)','Generic framework(0,20,0)',
                 #'Generic framework(0,0,2)','Generic framework(0,0,3)','Generic framework(0,0,4)','Generic framework(0,0,5)',
                # 'Generic framework(0,0,6)','Generic framework(0,0,7)','Generic framework(0,0,8)','Generic framework(0,0,9)',
                # 'Generic framework(0,0,10)','Generic framework(0,0,20)'
                 ]
clf_result = pd.DataFrame({#'threshold' : [],
                           #'depth': [],
                           #'n_estimators': [],
                           #'puf_seed' : [],
                           #'train_challenge_seed': [],
                           'PUF type': [],
                           'CRPs number':[],
                           'Test split Accuracy' : [],
                           'Test split F1' : [],
                           'Training time' : [],
                           'Testing time' : []
                           #'Cross_val Accuracy' : [],
                           #'Cross_val sd' : [],
                           #'Cross_val split F1' : [],
                           #'Cross_val F1 sd' : [],
                           })

#for (puf_seed, train_challenge_seed, test_challenge_seed) in zip(puf_seeds, train_challenge_seeds, test_challenge_seeds):
#for thresholds in threshold_val:
#for depth in depth_val:
    #for n_estimators in n_estimators_val:
for PUF_type in PUF_types:
        ### Set running start time ###
        start_time = datetime.now()
        
        ### Load data ###
        ### Load data ###
        if PUF_type == 'APUF':
            arbiter_puf = arbiter_PUF()
            data, data_label = arbiter_puf.load_data(68, 5000, 11, 123, 0)
            data_unseen, data_label_unseen = arbiter_puf.load_data(68, 5000, 11, 19, 0)
        elif PUF_type == '2 XOR-APUF':
            xor_puf = XOR_PUF()
            data, data_label = xor_puf.load_data(68, 5000, 2, 13,256,22,77,89,90, 11, 0)
            #data_unseen, data_label_unseen = xor_puf.load_data(68, 5000, 2, 13,256,22,77,89,90, 55, 0)
        elif PUF_type == '3 XOR-APUF':
            xor_puf = XOR_PUF()
            data, data_label = xor_puf.load_data(68, 5000, 3, 13,256,22,77,89,90, 11, 0)
            #data_unseen, data_label_unseen = xor_puf.load_data(68, 5000, 3, 13,256,22,77,89,90, 55, 0)
        elif PUF_type == '4 XOR-APUF':
            xor_puf = XOR_PUF()
            data, data_label = xor_puf.load_data(68, 5000, 4, 13,256,22,77,89,90, 11, 0)
            #data_unseen, data_label_unseen = xor_puf.load_data(68, 5000, 4, 13,256,22,77,89,90, 55, 0)
        elif PUF_type == '5 XOR-APUF':
            xor_puf = XOR_PUF()
            data, data_label = xor_puf.load_data(68, 5000, 5, 13,256,22,77,89,90, 11, 0)
            #data_unseen, data_label_unseen = xor_puf.load_data(68, 5000, 5, 13,256,22,77,89,90, 55, 0)
        elif PUF_type == '6 XOR-APUF':
            xor_puf = XOR_PUF()
            data, data_label = xor_puf.load_data(68, 5000, 6, 13,256,22,77,89,90, 11, 0)
            #data_unseen, data_label_unseen = xor_puf.load_data(68, 5000, 6, 13,256,22,77,89,90, 55, 0)
        elif PUF_type == 'FF-2-XOR-APUF':
            f1 = [5,12,26,19,33,49,51,7]
            d1 = [60,61,63,59,58,57,56,55]
            feedforward_puf = feedforward_PUF()
            data, data_label = feedforward_puf.load_data(68, 5000, 2, f1, d1, 256, 22, 77, 89, 90, 367, 23, 0)
            #data_unseen, data_label_unseen = feedforward_puf.load_data(68, 5000, 2, f1, d1, 256, 22, 77, 89, 90, 367, 334, 0)
        elif PUF_type == 'FF-3-XOR-APUF':
            f1 = [5,12,26,19,33,49,51,7]
            d1 = [60,61,63,59,58,57,56,55]
            feedforward_puf = feedforward_PUF()
            data, data_label = feedforward_puf.load_data(68, 5000, 3, f1, d1, 256, 22, 77, 89, 90, 367, 23, 0)
            #data_unseen, data_label_unseen = feedforward_puf.load_data(68, 5000, 3, f1, d1, 256, 22, 77, 89, 90, 367, 334, 0)
        elif PUF_type == 'FF-4-XOR-APUF':
            f1 = [5,12,26,19,33,49,51,7]
            d1 = [60,61,63,59,58,57,56,55]
            feedforward_puf = feedforward_PUF()
            data, data_label = feedforward_puf.load_data(68, 5000, 4, f1, d1, 256, 22, 77, 89, 90, 367, 23, 0)
            #data_unseen, data_label_unseen = feedforward_puf.load_data(68, 5000, 4, f1, d1, 256, 22, 77, 89, 90, 367, 334, 0)
        elif PUF_type == 'FF-5-XOR-APUF':
            f1 = [5,12,26,19,33,49,51,7]
            d1 = [60,61,63,59,58,57,56,55]
            feedforward_puf = feedforward_PUF()
            data, data_label = feedforward_puf.load_data(68, 5000, 5, f1, d1, 256, 22, 77, 89, 90, 367, 23, 0)
            #data_unseen, data_label_unseen = feedforward_puf.load_data(68, 5000, 5, f1, d1, 256, 22, 77, 89, 90, 367, 334, 0)
        elif PUF_type == 'FF-6-XOR-APUF':
            f1 = [5,12,26,19,33,49,51,7]
            d1 = [60,61,63,59,58,57,56,55]
            feedforward_puf = feedforward_PUF()
            data, data_label = feedforward_puf.load_data(68, 5000, 6, f1, d1, 256, 22, 77, 89, 90, 367, 23, 0)
            #data_unseen, data_label_unseen = feedforward_puf.load_data(68, 5000, 6, f1, d1, 256, 22, 77, 89, 90, 367, 334, 0)
        elif PUF_type == 'Generic framework(1,1,1)':
            #general_model = general_model()
            data, data_label = general_model.load_data(1, 1, 1, 0, 0, int(np.floor(5000/1)),50000)
            #data, data_label = shuffle(data, data_label)
            
            #general_model2 = general_model2()
            #data_unseen, data_label_unseen = general_model2.load_data(1, 1, 1, 0, 0, 5000)
            #data_unseen, data_label_unseen = shuffle(data_unseen, data_label_unseen)
        elif PUF_type == 'Generic framework(2,2,2)':
            #g1 = general_model()
            data, data_label = general_model.load_data(2, 2, 2, 0, 0, int(np.floor(5000/2)))
            #data, data_label = shuffle(data, data_label)
            
            #g2 = general_model2()
            #data_unseen, data_label_unseen = general_model2.load_data(4, 4, 4, 0, 0, 1250)
            #data_unseen, data_label_unseen = shuffle(data_unseen, data_label_unseen)
        elif PUF_type == 'Generic framework(3,3,3)':
            #g1 = general_model()
            data, data_label = general_model.load_data(3, 3, 3, 0, 0, int(np.floor(5000/3)))
            #data, data_label = shuffle(data, data_label)
            
            #g2 = general_model2()
            #data_unseen, data_label_unseen = general_model2.load_data(4, 4, 4, 0, 0, 1250)
            #data_unseen, data_label_unseen = shuffle(data_unseen, data_label_unseen)
        elif PUF_type == 'Generic framework(4,4,4)':
            #g1 = general_model()
            data, data_label = general_model.load_data(4, 4, 4, 0, 0, int(np.floor(5000/4)))
            #data, data_label = shuffle(data, data_label)
            
            #g2 = general_model2()
            #data_unseen, data_label_unseen = general_model2.load_data(4, 4, 4, 0, 0, 1250)
            #data_unseen, data_label_unseen = shuffle(data_unseen, data_label_unseen)
        elif PUF_type == 'Generic framework(5,5,5)':
            #g1 = general_model()
            data, data_label = general_model.load_data(5, 5, 5, 0, 0, int(np.floor(5000/5)))
            #data, data_label = shuffle(data, data_label)
        elif PUF_type == 'Generic framework(6,6,6)':
            #g1 = general_model()
            data, data_label = general_model.load_data(6, 6, 6, 0, 0, int(np.floor(5000/6)))
            #data, data_label = shuffle(data, data_label)
        elif PUF_type == 'Generic framework(10,10,10)':
            #g1 = general_model()
            data, data_label = general_model.load_data(10, 10, 10, 0, 0, int(np.floor(5000/10)))
            #data, data_label = shuffle(data, data_label)
        elif PUF_type == 'Generic framework(20,20,20)':
            #g1 = general_model()
            data, data_label = general_model.load_data(20, 20, 20, 0, 0, int(np.floor(5000/20)))
            #data, data_label = shuffle(data, data_label)
            
            #g2 = general_model2()
            #data_unseen, data_label_unseen = general_model2.load_data(4, 4, 4, 0, 0, 1250)
            #data_unseen, data_label_unseen = shuffle(data_unseen, data_label_unseen)
        elif PUF_type == 'Generic framework(2,0,0)':
            general_model = general_model()
            data, data_label = general_model.load_data(2, 0, 0, 0, 0, int(np.floor(15000/2)))
        elif PUF_type == 'Generic framework(3,0,0)':
            data, data_label = general_model.load_data(3, 0, 0, 0, 0, int(np.floor(15000/3)))
        elif PUF_type == 'Generic framework(4,0,0)':
            data, data_label = general_model.load_data(4, 0, 0, 0, 0, int(np.floor(15000/4)))
        elif PUF_type == 'Generic framework(5,0,0)':
            data, data_label = general_model.load_data(5, 0, 0, 0, 0, int(np.floor(15000/5)))
        elif PUF_type == 'Generic framework(6,0,0)':
            data, data_label = general_model.load_data(6, 0, 0, 0, 0, int(np.floor(15000/6)))
        elif PUF_type == 'Generic framework(7,0,0)':
            data, data_label = general_model.load_data(7, 0, 0, 0, 0, int(np.floor(15000/7)))
        elif PUF_type == 'Generic framework(8,0,0)':
            data, data_label = general_model.load_data(8, 0, 0, 0, 0, int(np.floor(15000/8)))
        elif PUF_type == 'Generic framework(9,0,0)':
            data, data_label = general_model.load_data(9, 0, 0, 0, 0, int(np.floor(15000/9)))
        elif PUF_type == 'Generic framework(10,0,0)':
            data, data_label = general_model.load_data(10, 0, 0, 0, 0, int(np.floor(15000/10)))
        elif PUF_type == 'Generic framework(20,0,0)':
            data, data_label = general_model.load_data(20, 0, 0, 0, 0, int(np.floor(15000/20)))

        elif PUF_type == 'Generic framework(0,2,0)':
            #general_model = general_model()
            data, data_label = general_model.load_data(0, 2, 0, 0, 0, int(np.floor(15000/2)))
        elif PUF_type == 'Generic framework(0,3,0)':
            data, data_label = general_model.load_data(0, 3, 0, 0, 0, int(np.floor(15000/3)))
        elif PUF_type == 'Generic framework(0,4,0)':
            data, data_label = general_model.load_data(0, 4, 0, 0, 0, int(np.floor(15000/4)))
        elif PUF_type == 'Generic framework(0,5,0)':
            data, data_label = general_model.load_data(0, 5, 0, 0, 0, int(np.floor(15000/5)))
        elif PUF_type == 'Generic framework(0,6,0)':
            data, data_label = general_model.load_data(0, 6, 0, 0, 0, int(np.floor(15000/6)))
        elif PUF_type == 'Generic framework(0,7,0)':
            data, data_label = general_model.load_data(0, 7, 0, 0, 0, int(np.floor(15000/7)))
        elif PUF_type == 'Generic framework(0,8,0)':
            data, data_label = general_model.load_data(0, 8, 0, 0, 0, int(np.floor(15000/8)))
        elif PUF_type == 'Generic framework(0,9,0)':
            data, data_label = general_model.load_data(0, 9, 0, 0, 0, int(np.floor(15000/9)))
        elif PUF_type == 'Generic framework(0,10,0)':
            data, data_label = general_model.load_data(0, 10, 0, 0, 0, int(np.floor(15000/10)))
        elif PUF_type == 'Generic framework(0,20,0)':
            data, data_label = general_model.load_data(0, 20, 0, 0, 0, int(np.floor(15000/20)))


        elif PUF_type == 'Generic framework(0,0,2)':
            #general_model = general_model()
            data, data_label = general_model.load_data(0, 0, 2, 0, 0, int(np.floor(15000/2)))
        elif PUF_type == 'Generic framework(0,0,3)':
            data, data_label = general_model.load_data(0, 0, 3, 0, 0, int(np.floor(15000/3)))
        elif PUF_type == 'Generic framework(0,0,4)':
            data, data_label = general_model.load_data(0, 0, 4, 0, 0, int(np.floor(15000/4)))
        elif PUF_type == 'Generic framework(0,0,5)':
            data, data_label = general_model.load_data(0, 0, 5, 0, 0, int(np.floor(15000/5)))
        elif PUF_type == 'Generic framework(0,0,6)':
            data, data_label = general_model.load_data(0, 0, 6, 0, 0, int(np.floor(15000/6)))
        elif PUF_type == 'Generic framework(0,0,7)':
            data, data_label = general_model.load_data(0, 0, 7, 0, 0, int(np.floor(15000/7)))
        elif PUF_type == 'Generic framework(0,0,8)':
            data, data_label = general_model.load_data(0, 0, 8, 0, 0, int(np.floor(15000/8)))
        elif PUF_type == 'Generic framework(0,0,9)':
            data, data_label = general_model.load_data(0, 0, 9, 0, 0, int(np.floor(15000/9)))
        elif PUF_type == 'Generic framework(0,0,10)':
            data, data_label = general_model.load_data(0, 0, 10, 0, 0, int(np.floor(15000/10)))
        elif PUF_type == 'Generic framework(0,0,20)':
            data, data_label = general_model.load_data(0, 0, 20, 0, 0, int(np.floor(15000/20)))


        
        
        data, data_unseen, data_label, data_label_unseen = train_test_split(data, data_label, test_size=.20)
        
        # print("data", data)
        # print("data_size", data.shape())
        
        #ML = LogisticRegression()
        #ML = DecisionTreeClassifier(criterion='entropy', max_depth=8)
        #ML = KNeighborsClassifier(n_neighbors=8)
        ML = svm.SVC(C=100, kernel='rbf')
        #ML = RandomForestClassifier(max_depth=8, n_estimators=100, criterion='entropy')
        ML.fit(data, data_label)
        
        ### Calculate training time ###
        end_time = datetime.now()
        #print('Training time: {}'.format(end_time - start_time))
        ss = StratifiedKFold(n_splits=5)
        ### Set testing start time ###
        test_start_time = datetime.now()
        results = ML.score(data, data_label)
        print('Training accuracy: {}%'.format(results*100))
        #cross_val = cross_val_score(xgboostModel, data_reduct, data_label,  cv=ss)
        #print("cross validation accuracy: %.2f%% (%.2f%%)" % (cross_val.mean()*100, cross_val.std()*100))
        
        data_unseen, data_label_unseen = shuffle(data_unseen, data_label_unseen)
        test_acc = ML.score(data_unseen, data_label_unseen)
        print("---------------------------------------")
        print('For unseen data')
        print('Testing accuracy: {}%'.format(test_acc*100))
        testingacc = ML.predict(data_unseen)
        cc = f1_score(data_label_unseen,testingacc)
        print(cc*100)
        #cross_val = cross_val_score(ML, data_unseen, data_label_unseen,  cv=ss)
        #print("cross validation accuracy: %.2f%% (%.2f%%)" % (cross_val.mean()*100, cross_val.std()*100))
        #cross_val_f1 = cross_val_score(ML, data_unseen, data_label_unseen, scoring="f1", cv=ss)
        #print("cross validation accuracy F1: %.2f%% (%.2f%%)" % (cross_val_f1.mean()*100, cross_val_f1.std()*100))
        
        ### Calculate testing time ###
        test_end_time = datetime.now()
        #print('Testing time: {}'.format(test_end_time - test_start_time))
        
        clf_result = clf_result.append({ #'threshold' : 0.01,
                                         #'depth': depth,
                                         #'n_estimators': n_estimators,
                                         #'puf_seed' : 11,
                                         #'train_challenge_seed': 123,
                                         #'test_challenge_seed': 19,
                                         'PUF type':PUF_type,
                                         'CRPs number':5000,
                                         'Test split Accuracy' : test_acc*100,
                                         'Test split F1' : cc*100,
                                         'Training time' : end_time - start_time,
                                         'Testing time' : test_end_time - test_start_time,
                                         #'Cross_val Accuracy' : cross_val.mean()*100,
                                         #'Cross_val sd' : cross_val.std()*100,
                                         #'Cross_val split F1' : cross_val_f1.mean()*100,
                                         #'Cross_val F1 sd' : cross_val_f1.std()*100
                                         },  ignore_index=True)
        
        #clf_result.to_csv(r'C:\Users\weber\OneDrive\Desktop\Dissertation\XGBoost\{}.csv'.format(puf_seed))
clf_result.to_csv(r'D:\codes\Dissertation\XGBoost\SVM_all_PUF.csv')