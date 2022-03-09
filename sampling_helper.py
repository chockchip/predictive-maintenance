from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier\
,BalancedRandomForestClassifier, RUSBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

class SamplingHelper():

    RANDOM_STATE = 123

    def check_df_contain_type(df, types):
        result = True
        for type in df.dtypes.items():
            if type[1] not in types:
                result = False
        return result

    def under_sampling(x, y, sampling_strategy=1, random_state=123):
    
        undersampler = RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy)
        x_rs, y_rs = undersampler.fit_resample(x, y)
        return(x_rs, y_rs)

    def over_sampling(x, y, sampling_strategy=1, random_state=123):
    
        oversampler = RandomOverSampler(random_state=SamplingHelper.RANDOM_STATE, sampling_strategy=sampling_strategy)
        x_rs, y_rs = oversampler.fit_resample(x, y)
        return(x_rs, y_rs)

    def smote_sampling(x, y, random_state=123):
    
        # The input type should be int64, float64
        # correctType = SamplingHelper.check_df_contain_type(x, ['int64', 'float64'])
    
        # if not correctType:
        #     raise Exception("Some data don't have type int64 or float64")
        
        smote = SMOTE(random_state=SamplingHelper.RANDOM_STATE)
        x_rs, y_rs = smote.fit_resample(x, y)
        return (x_rs, y_rs)

    def smotetomek_sampling(x, y, random_state=123):
    
        # The input type should be int64, float64
        # correctType = SamplingHelper.check_df_contain_type(x, ['int64', 'float64'])
        
        # if not correctType:
        #     raise Exception("Some data don't have type int64 or float64")
        
        smote_tomek = SMOTETomek(random_state=SamplingHelper.RANDOM_STATE)
        x_rs, y_rs = smote_tomek.fit_resample(x, y)
        return (x_rs, y_rs)

    def ratio_sampling(x, y, negative_size, positive_size):

        # The input type should be int64, float64
        correctType = SamplingHelper.check_df_contain_type(x, ['int64', 'float64'])
        
        if not correctType:
            raise Exception("Some data don't have type int64 or float64")
        
        count_class_0 = negative_size
        count_class_1 = positive_size
        
        pipe = make_pipeline(SMOTE(sampling_strategy={1: count_class_1}), NearMiss(sampling_strategy={0: count_class_0}))
        x_rs, y_rs = pipe.fit_resample(x, y)
        return (x_rs, y_rs)