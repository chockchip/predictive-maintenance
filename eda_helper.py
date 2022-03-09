# -------------------------------------------------------------------
# Title        : Exploratory Data Analysis (EDA) Helper
# Description  :
# Writer       : Watcharapong Wongrattanasirikul
# Created date : 27 Jul 2021
# Updated date : -
# Version      : 0.0.2
# Remark       : Update description in each funtion
# -------------------------------------------------------------------
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np  
from numpy import unique
from numpy import where
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import RFE
import pandas as pd 

class EdaHelper():

    # --------------------------------------------------------------------------
    # description: Create new dataframe from 2 list, array or dataframe
    # arguments  : x(list, array, df) -> data that need to convert to dataframe
    #            : y(list, array, df) -> data that need to convert to dataframe
    # return     : df(dataframe)      -> data that merge from 2 input
    # --------------------------------------------------------------------------
    def get_dataframe_xy(x, y):
        df = pd.concat([pd.DataFrame(data=y), pd.DataFrame(data=x)], axis=1, sort=False )
        return df

    # --------------------------------------------------------------------------
    # description: Plot the bar chart of missing percentage of each feature
    # arguments  : df(dataframe) -> data to plot the missing
    # --------------------------------------------------------------------------
    def plot_missing_chart(df):
        missing_percentage = round((df.isna().sum()/len(df)) * 100, 2)
        missing_percentage = missing_percentage[missing_percentage > 0]
        missing_percentage.sort_values(ascending=False,inplace=True)
        missing_percentage.plot.bar(figsize=(22,8))

        plt.grid(True)
        plt.title('Missing Data', fontsize=18)
        plt.ylabel("Proportion Missing (%)", fontsize=12)
        sns.despine()

    # --------------------------------------------------------------------------
    # description: Plot the scatter plot of 2 dimension and sepatate class by color
    # arguments  : X(dataframe) -> feature columns (x) should have at leas 2 features
    # --------------------------------------------------------------------------
    def plot_data_by_class(X, y):
        n_classess = len(unique(y))
        for class_value in range(n_classess):
            row_ix = where(y == class_value)[0]
            plt.scatter(X.iloc[row_ix, 0], X.iloc[row_ix, 1], label=str(class_value), alpha=0.3)
        plt.legend()
        plt.show()
        
    # --------------------------------------------------------------------------
    # description: Get multicollinearity score
    # arguments  : df(dataframe) -> data that need to find multicollinearity score
    # return     : vif_data(dataframe) -> feature with vif score
    #
    # vif score  : <5 Low
    #            : 5-9 Medium
    #            : >10 High
    # --------------------------------------------------------------------------
    def get_multicollinearity_score(df):
        vif_data = pd.DataFrame()
        vif_data["feature"] = df.columns
  
        # calculating VIF for each feature
        vif_data["VIF"] = [variance_inflation_factor(df.values, i)
                                for i in range(len(df.columns))]
        
        return vif_data

    def plot_anormaly_score(df, column):
        # Train IsolationForest
        isolation_forest = IsolationForest(n_estimators=100)
        isolation_forest.fit(df[column].values.reshape(-1, 1))

        # Store sale data in Numpy. ?? Why store data like this
        xx = np.linspace(df[column].min(), df[column].max(), len(df)).reshape(-1, 1)

        # Computed the anomaly score for each observation. -> It compute the mean anomaly score
        anomaly_score = isolation_forest.decision_function(xx)

        # Classify outlier/non-outlier
        outlier = isolation_forest.predict(xx)

        plt.figure(figsize=(10,4))
        plt.plot(xx, anomaly_score, label='anomaly score')
        plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 
                        where=outlier==-1, color='r', 
                        alpha=.4, label='outlier region')
        plt.legend()
        plt.ylabel('anomaly score')
        plt.xlabel(str(column))
        plt.show()

    # --------------------------------------------------------------------------
    # description: Get list of columns that have duplicate data between columns
    # arguments  : df(dataframe) -> data to check duplicate columns
    # return     : duplicateColumnNames(list) -> list of column name that data duplicated
    # --------------------------------------------------------------------------
    def get_duplicate_columns(df):
        duplicateColumnNames = set()
    
        for x in range(df.shape[1]):
            
            # Take column at xth index.
            col = df.iloc[:, x]
            
            # Iterate through all the columns in
            # DataFrame from (x + 1)th index to
            # last index
            for y in range(x + 1, df.shape[1]):
                
                # Take column at yth index.
                otherCol = df.iloc[:, y]
                
                # Check if two columns at x & y
                # index are equal or not,
                # if equal then adding 
                # to the set
                if col.equals(otherCol):
                    duplicateColumnNames.add(df.columns.values[y])
                    
        # Return list of unique column names 
        # whose contents are duplicates.
        return list(duplicateColumnNames)

    # --------------------------------------------------------------------------
    # description: Get the feature with importance score
    # arguments  : df(datafrane)   -> data that want to get the importance score
    #            : label(string)   -> column name of the prediction class
    #            : score(float)    -> filter only features that has score more 
    #                                 than setting score
    # --------------------------------------------------------------------------
    def get_feature_importance(df, label, score=0) :
        x = df.drop([label], axis=1)
        y = df[label]
        model  =  RandomForestRegressor(n_estimators=100)
        model.fit(x,y)
        
        # check type value 
        feature_score = model.feature_importances_
        
        df_feature_importance =  pd.DataFrame(feature_score, index = list(x.columns), columns= ['score']).sort_values(by = 'score' , ascending = False)
        
        if score >= 0:
            df_feature_importance = df_feature_importance[df_feature_importance['score'] > score]
        
        return df_feature_importance

    # --------------------------------------------------------------------------
    # description: Get importance features by RFE
    # arguments  : df(dataframe)) -> data that need to get the importance features
    #            : label(string)  -> column name of prediction class (y)
    #            : number(int)    -> number of importance feature
    # return     : df_RFE_results(dataframe) -> features with selected and ranking
    # --------------------------------------------------------------------------
    def get_feature_by_rfe(df, label, number, model):

        y = df[label]
        x = df.drop([label], axis=1)

        rfe = RFE(model, number)
        fit = rfe.fit(x, y)

        df_RFE_results = []
        for i in range(x.shape[1]):
            df_RFE_results.append(
                {      
                    'Feature_names': x.columns[i],
                    'Selected':  rfe.support_[i],
                    'RFE_ranking':  rfe.ranking_[i],
                }
            )

        df_RFE_results = pd.DataFrame(df_RFE_results)
        df_RFE_results.index.name='Columns'
        return df_RFE_results

    #! Wait for check how does it work
    def get_duplicate_data_in_column(df):

        df_duplicate = pd.DataFrame(columns= ['column','duplicate'])

        for column in df.columns:
            row = {'column': column, 'duplicate':df.pivot_table(columns=column, aggfunc='size').max()}
            df_duplicate = df_duplicate.append(row, ignore_index=True)

        return df_duplicate
