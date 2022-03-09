# -------------------------------------------------------------------
# Title        : Imputation Helper
# Description  :
# Writer       : Watcharapong Wongrattanasirikul
# Created date : 27 Jul 2021
# Updated date : -
# Version      : 0.0.1
# Remark       : Helper to impute data
# -------------------------------------------------------------------
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import numpy as np
from enum import Enum

class imputation_strategy(Enum):
    mean = 1
    median = 2
    mode = 3
   
class ImputeHelper():

    # Mapping strategy of statistic imputation
    strategy_mapper = {
        imputation_strategy.mean: 'mean'
        ,imputation_strategy.median: 'median'
        ,imputation_strategy.mode: 'most_frequent'
    }

    # -------------------------------------------------------------------
    # description: Impute missing value by statistic
    # arguments  : x(dataframe) -> data that need to impute 
    #            : strategy(enum) -> strategy of statistic imputation
    # return     : x_trans(dataframe) -> data that already impute
    # -------------------------------------------------------------------
    def impute_statistic(x, strategy):
        
        impute_strategy = ImputeHelper.strategy_mapper[strategy]
 
        statistic_imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strategy)
        x_trans = statistic_imputer.fit_transform(x)

        return x_trans

    # -------------------------------------------------------------------
    # description: Impute missing value by knn
    # arguments  : x(dataframe) -> data that need to impute 
    #            : number_neighbor(int) -> Number of neighbor on Knn 
    # return     : x_trans(dataframe) -> data that already impute
    # -------------------------------------------------------------------
    def impute_knn(x, number_neighbor=5):

        knn_imputor = KNNImputer(n_neighbors= number_neighbor)
        x_trans = knn_imputor.fit_transform(x)

        return x_trans