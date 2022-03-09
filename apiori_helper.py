from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
import pandas as pd
import numpy as np
from scipy.sparse import data

class ApioriHelper():

    # -------------------------------------------------------------------
    # description: Clear Nan data in the dataframe. It will make the length
    #              of list in each line aren't equal
    # arguments  : basket_data(dataframe) -> data
    #            : has_header(bool) -> Is data has column
    # return     : dataset(list) -> data without NaN
    # -------------------------------------------------------------------
    @staticmethod
    def cleanup_nan(basket_data, has_herader=False):

        if has_herader is True:
            basket_data = basket_data.iloc[1: , :]

        dataset = []
        for i in range(0, basket_data.shape[0]):
            temp = []
            for j in range(0, basket_data.shape[1]):
                if basket_data.iloc[i, j] is not np.nan and basket_data.iloc[i, j] != '' :
                    b = basket_data.iloc[i, j]
                    temp.append(basket_data.iloc[i,j])
            dataset.append(temp)

        return dataset

    # -------------------------------------------------------------------
    # description: Create onehot encoder from the data table
    # arguments  : dataset(list) -> data table
    # return     : df(dataframe) -> data table with onehot encoder
    # -------------------------------------------------------------------
    @staticmethod
    def convert_list_to_onehot_encoder(dataset):
        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        return df

    # -------------------------------------------------------------------
    # description: Create frequent itemsets
    # arguments  : df(dataframe) -> data table with onehot endcoder
    #            : min_support(float) -> support for decided
    # return     : frequent_itemsets(dataframe) -> frequenct itemsets 
    # -------------------------------------------------------------------
    @staticmethod
    def get_apriori_frequent_itemsets(df, min_support=0.05, use_colnames=True):
        frequent_itemsets = apriori(df, min_support, use_colnames)
        return frequent_itemsets

    # -------------------------------------------------------------------
    # description: Create association rules
    # arguments  : frequent_itemsets(dataframe) -> frequenct itemsets 
    #            : metric(string) -> metric that use for measure
    #            : min_threshold(integer) -> min threshold for create rules
    # return     : rules(dataframe) -> association rules 
    # -------------------------------------------------------------------
    @staticmethod
    def get_association_rules(frequent_itemsets, metric='lift', min_threshold=1):
        rules = association_rules(frequent_itemsets, metric, min_threshold)
        return rules

