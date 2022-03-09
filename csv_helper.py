import os,sys,inspect
from types import prepare_class
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import csv
import pandas as pd

class CsvHelper():

    '''
    When use Pandas to read CSV. It use number of the first column to be 
    maximum expected columns of each rows then it will cause the tokenize error
    becuase some line has the number of columns more than expected. This class
    will help for read_csv with maximum columns in the file
    '''

    # -------------------------------------------------------------------
    # description: Read complete csv
    # arguments  : path(string) -> path of the csv file
    #            : separator(string) -> The separator that use in file
    # return     : df(dataframe) -> data from the csv
    # -------------------------------------------------------------------
    @staticmethod
    def read_complete_csv(path, separator=','):
        with open(path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=separator)
            exsiting_data = [line for line in csv_reader]
            
            df = pd.DataFrame(exsiting_data)

            return df

    # -------------------------------------------------------------------
    # description: Get number of the maximum columns of the csv file
    # arguments  : path(string) -> path of csv
    # return     : max_columns(int) -> number of maximum columns (integer)
    # -------------------------------------------------------------------
    @staticmethod
    def get_maximum_columns(path, separator=','):
        max_columns = 0
        with open(path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=separator)

            for line in csv_reader:
                if len(line) > max_columns:
                    max_columns = len(line)

            return max_columns

    # -------------------------------------------------------------------
    # description: Generate csv file from dataframe
    # arguments  : df(dataframe) -> data 
    #            : path(string) -> path of csv
    #            : header(list) -> header of each columns
    # -------------------------------------------------------------------
    @staticmethod
    def generate_csv_from_dataframe(df, path, header=None):
        with open(path, 'a') as f:
            w = csv.writer(f)
            
            if header is not None:
                w.writerow(header)
        
            w.writerows(df.values.tolist())