# -------------------------------------------------------------------
# Title        : Machine Learning Helper
# Description  :
# Writer       : Watcharapong Wongrattanasirikul
# Created date : 27 Jul 2021
# Updated date : -
# Version      : 0.0.1
# Remark       : Helper to make the model from machine learning
# -------------------------------------------------------------------
from pandas.core.indexing import _ScalarAccessIndexer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier, BaggingRegressor, GradientBoostingClassifier,BaggingClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier , LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from enum import Enum

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sampling_helper import SamplingHelper

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from IPython.display import display

import scikitplot as skplt
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# Declare list of machine learning technique that available in this helper
class model_type(Enum):
    svm = 1
    logistics = 2
    decisiontree = 3
    xgboots = 4
    gaussian_nb = 5
    sgd = 6
    randomforest = 7
    knn = 8

# Declare list of machine learning technique that available in this helper
class resampler_type(Enum):
    over = 1
    under = 2
    smote = 3
    smote_tomek = 4

# Declare list of scaling technique that available in this helper
class scaler_type(Enum):
    na = 1
    standardscaler = 2
    normalizer = 3

class MlHelper():


    # --------------------------------------------------------------------------
    # description: Split data to be train and test set
    # arguments  : df(dataframe)    -> data that need to be split train/test
    #            : label(string)    -> predict class
    #            : test_size(float) -> Size(%) of test data 
    # return     : x_train(array)   -> x of train set
    #            : y_train(array)   -> y of train set
    #            : x_test(array)    -> x of test set
    #            : y_test(array)    -> y of test set
    # --------------------------------------------------------------------------        
    def split_train_test(df ,label, test_size=0.2, RANDOM_STATE=123):
    
        # Split train-test data
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
        for train_index, test_index in split.split(df, df[label]):
            train_set = df.loc[train_index]
            test_set = df.loc[test_index]
            
        # Split label, feature
        y_train = train_set[label]
        x_train = train_set.drop([label], axis=1)
        y_test = test_set[label]
        x_test = test_set.drop([label], axis=1)
          
        return (x_train, y_train, x_test, y_test)
    
    # --------------------------------------------------------------------------
    # description: Split data to be predition class (y) and feature (x)
    # arguments  : df(dataframe) -> data that need to split
    #            : label(strin)  -> column name of the prediction class (y)
    # return     : x(dataframe)  -> features (x)
    #            : y(dataframe)  -> prediciton class (y)
    # --------------------------------------------------------------------------
    def split_label_feature(df, label):
        x = df.drop([label], axis=1)
        y = df[label]

        return (x, y)

    # --------------------------------------------------------------------------
    # description: Create the SVM model
    # arguments  : x_train(array)    -> features (x) of train data
    #            : y_train(array)    -> predict class (y) of train data
    #            : x_test(array)     -> features (x) of test data
    # return     : svm_model(model)  -> machine learning model
    #            : y_pred(array)     -> predicted class (y) of test data
    #            : y_prob(array)     -> predicted propability (y) of test data
    # --------------------------------------------------------------------------
    def get_svm(x_train, y_train, x_test, y_test):
    
        clf = SVC(kernel='linear',probability=True)
        svm_model = clf.fit(x_train, y_train)

        y_pred = svm_model.predict(x_test)
        y_prob = svm_model.predict_proba(x_test)
        
        return (svm_model, y_pred, y_prob)

    # --------------------------------------------------------------------------
    # description: Create the LogisticRegression model
    # arguments  : x_train(array)    -> features (x) of train data
    #            : y_train(array)    -> predict class (y) of train data
    #            : x_test(array)     -> features (x) of test data
    # return     : logistic_model(model)  -> machine learning model
    #            : y_pred(array)     -> predicted class (y) of test data
    #            : y_prob(array)     -> predicted propability (y) of test data
    # --------------------------------------------------------------------------
    def get_logistics(x_train, y_train, x_test, y_test):
    
        clf = LogisticRegression(solver='liblinear', class_weight='balanced')
        logistic_model = clf.fit(x_train, y_train)
        
        y_pred = logistic_model.predict(x_test)
        y_prob = logistic_model.predict_proba(x_test)
        
        return (logistic_model, y_pred, y_prob)

    # --------------------------------------------------------------------------
    # description: Create the XGboots model
    # arguments  : x_train(array)    -> features (x) of train data
    #            : y_train(array)    -> predict class (y) of train data
    #            : x_test(array)     -> features (x) of test data
    #            : y_test(array)     -> predict class (y) of test data
    # return     : svm_model(model)  -> machine learning model
    #            : y_pred(array)     -> predicted class (y) of test data
    #            : y_prob(array)     -> predicted propability (y) of test data
    # --------------------------------------------------------------------------
    def get_xgboots(x_train, y_train, x_test, y_test):
        
        #! Only this model that use y_test to fit model why?
        clf =  xgb.XGBClassifier(objective='binary:logistic', eval_metric="auc")
        xgboot_model = clf.fit(x_train, y_train, early_stopping_rounds=5, eval_set=[(x_test, y_test)])

        y_pred =  xgboot_model.predict(x_test)
        y_prob = xgboot_model.predict_proba(x_test)
        
        return (xgboot_model, y_pred, y_prob)   
    
    # --------------------------------------------------------------------------
    # description: Create the GaussianNaviveBayes model
    # arguments  : x_train(array)    -> features (x) of train data
    #            : y_train(array)    -> predict class (y) of train data
    #            : x_test(array)     -> features (x) of test data
    # return     : gaussian_nb_model(model)  -> machine learning model
    #            : y_pred(array)     -> predicted class (y) of test data
    #            : y_prob(array)     -> predicted propability (y) of test data
    # --------------------------------------------------------------------------
    def get_gaussian_naive_bayes(x_train, y_train, x_test, y_test):
    
        clf =  GaussianNB()
        gaussian_nb_model = clf.fit(x_train, y_train)
        
        y_pred = gaussian_nb_model.predict(x_test) 
        y_prob =  gaussian_nb_model.predict_proba(x_test)
        
        return (gaussian_nb_model, y_pred, y_prob)  

    # --------------------------------------------------------------------------
    # description: Create the SGDClassifier model
    # arguments  : x_train(array)    -> features (x) of train data
    #            : y_train(array)    -> predict class (y) of train data
    #            : x_test(array)     -> features (x) of test data
    # return     : sgd_model(model)  -> machine learning model
    #            : y_pred(array)     -> predicted class (y) of test data
    #            : y_prob(array)     -> predicted propability (y) of test data
    # --------------------------------------------------------------------------
    def get_sgd(x_train, y_train, x_test, y_test):
    
        clf =  SGDClassifier(loss='modified_huber', shuffle=True,random_state=101)
        sgd_model = clf.fit(x_train, y_train)
        
        y_pred = sgd_model.predict(x_test)
        y_prob  =  sgd_model.predict_proba(x_test)
        return (sgd_model, y_pred, y_prob)   

    # --------------------------------------------------------------------------
    # description: Create the RandomForestClassifier model
    # arguments  : x_train(array)    -> features (x) of train data
    #            : y_train(array)    -> predict class (y) of train data
    #            : x_test(array)     -> features (x) of test data
    # return     : randomforest_model(model)  -> machine learning model
    #            : y_pred(array)     -> predicted class (y) of test data
    #            : y_prob(array)     -> predicted propability (y) of test data
    # --------------------------------------------------------------------------
    def get_randomforest(x_train, y_train, x_test, y_test):
    
        clf =RandomForestClassifier(n_estimators=100, random_state = 13)

        #Train the model using the training sets y_pred=clf.predict(X_test)
        randomforest_model = clf.fit(x_train,y_train)
        
        y_pred = randomforest_model.predict(x_test)        
        y_prob  =  randomforest_model.predict_proba(x_test)
        return (randomforest_model, y_pred, y_prob)  

    # --------------------------------------------------------------------------
    # description: Create the KNeighborsClassifier model
    # arguments  : x_train(array)    -> features (x) of train data
    #            : y_train(array)    -> predict class (y) of train data
    #            : x_test(array)     -> features (x) of test data
    # return     : knn_model(model)  -> machine learning model
    #            : y_pred(array)     -> predicted class (y) of test data
    #            : y_prob(array)     -> predicted propability (y) of test data
    # --------------------------------------------------------------------------
    def get_knn(x_train, y_train, x_test, y_test):

        clf =  KNeighborsClassifier(n_neighbors=3)
        knn_model = clf.fit(x_train,y_train)
        
        y_pred = knn_model.predict(x_test)
        y_prob = knn_model.predict_proba(x_test)
        return(knn_model, y_pred, y_prob)  

    # --------------------------------------------------------------------------
    # description: Create the DecisionTreeClassifier model
    # arguments  : x_train(array)    -> features (x) of train data
    #            : y_train(array)    -> predict class (y) of train data
    #            : x_test(array)     -> features (x) of test data
    # return     : decisiontree_model(model)  -> machine learning model
    #            : y_pred(array)     -> predicted class (y) of test data
    #            : y_prob(array)     -> predicted propability (y) of test data
    # --------------------------------------------------------------------------
    def get_decisiontree(x_train, y_train, x_test, y_test):

        min_leaf = 1
        clf = DecisionTreeClassifier(max_depth=10, random_state=101, max_features = None, min_samples_leaf = min_leaf)
        decisiontree_model = clf.fit(x_train,y_train)
        
        y_pred = decisiontree_model.predict(x_test)
        y_prob  =  decisiontree_model.predict_proba(x_test)
        return(decisiontree_model, y_pred, y_prob)  

    # --------------------------------------------------------------------------
    # description: Create the basic deep learning sequential model
    # arguments  : x_train(array)    -> features (x) of train data
    #            : y_train(array)    -> predict class (y) of train data
    #            : x_test(array)     -> features (x) of test data
    # return     : deep_model(model)  -> deep learning model
    #            : y_pred(array)     -> predicted class (y) of test data
    #            : y_prob(array)     -> predicted propability (y) of test data
    # --------------------------------------------------------------------------
    def get_deep(x_train, y_train, x_test, y_test):
        ### created model  ###
        model = Sequential()  
        model.add(Dense(256, kernel_initializer='uniform',activation='relu',input_dim= x_train.shape[1]))
        #model.add(Dense(256,activation='relu'))
        model.add(Dense(128, kernel_initializer='uniform',activation='relu'))
        model.add(Dense(64, kernel_initializer='uniform',activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer='uniform',activation='sigmoid'))
        ### add complie###

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        deep_model = model.fit(x = x_train ,y = y_train,epochs = 200,batch_size=512) #,validation_data=(x_test ,y_test)
    
        y_pred = model.predict(x_test)
        y_prob = np.NaN

        return(deep_model, y_pred, y_prob)

    # This part use for create the report to trial multiple model, scale, etc.

    # Mapping of machine learning models
    model_mapper = {model_type.svm: get_svm,
            model_type.logistics: get_logistics,
            model_type.decisiontree: get_decisiontree,
            model_type.xgboots: get_xgboots,
            model_type.gaussian_nb: get_gaussian_naive_bayes,
            model_type.sgd: get_sgd,
            model_type.randomforest: get_randomforest,
            model_type.knn: get_knn}

    # Mapping of resampler methods
    resampler_mapper = {resampler_type.over: SamplingHelper.over_sampling, 
                   resampler_type.under: SamplingHelper.under_sampling,
                   resampler_type.smote: SamplingHelper.smote_sampling,
                   resampler_type.smote_tomek: SamplingHelper.smotetomek_sampling}

    # <apping of scaler methods
    scaler_mapper = {scaler_type.na: None,
                     scaler_type.normalizer: Normalizer(),
                     scaler_type.standardscaler: StandardScaler()}

    # --------------------------------------------------------------------------
    # description: Create the DecisionTreeClassifier model
    # arguments  : x_train(array)    -> features (x) of train data
    #            : y_train(array)    -> predict class (y) of train data
    #            : x_test(array)     -> features (x) of test data
    # return     : decisiontree_model(model)  -> machine learning model
    #            : y_pred(array)     -> predicted class (y) of test data
    #            : y_prob(array)     -> predicted propability (y) of test data
    # --------------------------------------------------------------------------
    def scale_data(x , scaler_type):
        scaler_fn = MlHelper.scaler_mapper[scaler_type]
        x_scale = scaler_fn.fit_transform(x)
        return x_scale

    # --------------------------------------------------------------------------
    # description: Create the machine leaning model base on specific condition
    # arguments  : df(dataframe)     -> Data to create the machine learning
    #            : label(string)     -> column name ofpredict class (y) 
    #            : scaler(model)     -> scaler model
    #            : resampler(model)  -> machine learning model
    #            : model(model)      -> machine learning model 
    # return     : clf(model)        -> machien learning model after training
    #            : data(dataframe)   -> Report of prediction result
    # --------------------------------------------------------------------------
    def create_model(df, label, scaler, resampler, model):
     
        # 0. Prepare report
        df_report = pd.DataFrame(columns = ['Model','Sampling','Scaler','tp','tn','fn','fp','Accuracy','AUC','Precision','Recall','F1_Score',])
        
        # 1. Split data
        x_train, y_train, x_test, y_test = MlHelper.split_train_test(df, label)
        
        # 2. Scale features
        if scaler is not None:
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.fit_transform(x_test)
        
        # 3. Re-sampling train data to reduce impact of imbalance
        x_train_rs, y_train_rs = resampler(x_train, y_train)
        
        # 4. Fit model
        clf, y_pred, y_prob = model(x_train_rs, y_train_rs, x_test, y_test)
        
        # 5. Generate data
        tn, fp, fn, tp  = confusion_matrix(y_test, y_pred).ravel()
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)
        precision = tp/(tp+fp) 
        recall = tp/(tp+fn)
        f1_score = (2*precision*recall)/(precision+recall)
        
        # 6. Add data to report
        # data = {'Model': model.__name__[4:], 'Sampling': resampler.__name__,'Scaler':scaler, 'tp': tp, 'tn': tn, 'fn': fn, 'fp': fp, 
        #         'Accuracy': accuracy, 'AUC':auc_score, 'Precision':precision, 'Recall':recall, 
        #         'F1_Score': f1_score}

        data = [model.__name__[4:], resampler.__name__, scaler, tp, tn, fn, fp, accuracy, auc_score, precision, recall, f1_score]
        
        df_report.loc[len(df_report)] = data
        
        return (clf, data)

    # --------------------------------------------------------------------------
    # description: Create report of machine leaning models
    # arguments  : df(dataframe)     -> Data to create the machine learning
    #            : label(string)     -> column name ofpredict class (y) 
    #            : scaler(list)      -> list of scaler model
    #            : resampler(list)   -> list of machine learning model
    #            : model(list)       -> list of machine learning model 
    # return     : df_report(dataframe) -> Summary report
    # --------------------------------------------------------------------------
    def generate_report(df, label, scalers, resamplers, models):

        df_report = pd.DataFrame(columns = ['Model','Sampling','Scaler','tp','tn','fn','fp','Accuracy','AUC','Precision','Recall','F1_Score',])
        
        for model in models:
            model_fn = MlHelper.model_mapper[model]
            
            for resampler in resamplers:
                resampler_fn = MlHelper.resampler_mapper[resampler]
                
                for scaler in scalers:
                    scaler_fn = MlHelper.scaler_mapper[scaler]
                    
                    try:
                        _, data = MlHelper.create_model(df, label, scaler_fn, resampler_fn, model_fn)
                        df_report.loc[len(df_report)] = data
                        print(f'Complete on model: {model_fn.__name__[4:]}, resampler: {resampler_fn.__name__}, scaler: {scaler}')
                    except Exception as e:
                        print(f'Error occur on model: {model_fn.__name__[4:]}, resampler: {resampler_fn.__name__}, scaler: {scaler}')
                        print('*'*50)
                        print(e)
                        print('*'*50)
        return df_report

    # --------------------------------------------------------------------------
    # description: Plot the RoC curve to eveluate the prediction result
    # arguments  : model(model)     -> machine learning model
    #            : x_test(array)    -> feature (x)
    #            : y_test(array)    -> prediction class (y)
    # --------------------------------------------------------------------------
    def plot_roc_curve(model, x_test, y_test):
        y_prob = model.predict_proba(x_test)
        skplt.metrics.plot_roc(y_test, y_prob, figsize=(8,8))
        plt.show()

    # --------------------------------------------------------------------------
    # description: Drop unimportance feature
    # arguments  : df(datafrane)   -> data that want to drop features
    #            : importance_features(list)   -> columns name of importance feature
    #            : label(string)   -> column name of prediction class (y)
    # --------------------------------------------------------------------------
    def drop_unimportance_feature(df, importance_features, label):
        
        all_features = list(df.columns)
        importance_feature = list(importance_features.index)
        
        all_features.remove(label)
        
        for feature in importance_feature:
            all_features.remove(feature)
        
        df_result = df.drop(all_features, axis=1)
        
        return df_result

    # --------------------------------------------------------------------------
    # description: Get the RoC score from prediction
    # arguments  : y_pred(datafrane)   -> prediction class from model (y hat)
    #            : y_test(datafrane)   -> actual prediction class (y)
    # --------------------------------------------------------------------------
    def get_roc_auc_score(y_pred, y_test):
        roc_auc = roc_auc_score(y_test, y_pred)
        print("Roc_Auc: %.2f%%" % (roc_auc * 100.0))

    # --------------------------------------------------------------------------
    # description: Get the confusion matrix from prediction
    # arguments  : y_pred(datafrane)   -> prediction class from model (y hat)
    #            : y_test(datafrane)   -> actual prediction class (y)
    # --------------------------------------------------------------------------
    def get_confusion_matrix(y_test, y_pred):
        matrix = confusion_matrix(y_test, y_pred, labels=[1,0])
        print('Confusion matrix : \n',matrix, '\n\n', 'row:actual', '\n', 'col:predict')

    # --------------------------------------------------------------------------
    # description: Get the decision score from prediction
    # arguments  : y_pred(datafrane)   -> prediction class from model (y hat)
    #            : y_test(datafrane)   -> actual prediction class (y)
    # --------------------------------------------------------------------------
    def get_decision_score(y_pred, y_test):
        print(classification_report(y_test, y_pred))

    # --------------------------------------------------------------------------
    # description: Decorate the report that create by generate_report
    # arguments  : df(dataframe)    -> Report of machine learning performance
    # --------------------------------------------------------------------------
    def display_metrics(df):
        return df.style.bar(subset=['Accuracy'], color='lightskyblue')\
                        .bar(subset=['AUC'], color='lightsalmon')\
                        .bar(subset=['Precision','Recall'], color='lightgray')\
                        .bar(subset=['F1_Score'], color='lightgreen')\
                        .format({'Accuracy':'{:.2%}','AUC':'{:.2%}','Precision':'{:.2%}','Recall':'{:.2%}','F1_Score':'{:.2%}'})