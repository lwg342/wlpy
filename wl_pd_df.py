# -> Created on 26 November 2020
# -> Author: Weiguang Liu
# %%
import numpy as np 
import  pandas as pd
# %%


class DFManipulation():
    """
    Provides methods to manipulate a Pandas Dataframe. \n
    See the wiki page for list of methods.
    """
    def __init__(self, DF):
        self.DF = DF

    def rolling_window(self, window_size, step, min_size=1):
        """
        Input: DF, one Pandas Dataframe\n
        window_size : the length of each window\n
        step: the distance between the starts of two consecutive windows\n
        min_size: the mimimum number of observations in one window\n
        Return: a List of DF's        
        """
        DF = self.DF
        date_list = DF.index
        starts = []
        ends = []
        T = len(date_list)
        for i in range(0, T, step):
            if i + min_size > T-1:
                # if i >=20 :
                break
            elif i + window_size > T-1:
                starts = starts + [date_list[i]]
                ends = ends + [date_list[-1]]
            else:
                starts = starts + [date_list[i]]
                ends = ends + [date_list[i + window_size]]
            # print(i)
        num_of_windows = len(starts)
        DF_rolling = [DF.loc[(DF.index >= starts[i]) * (DF.index < ends[i])]
                    for i in range(num_of_windows)]
        return DF_rolling
# %%


class clean_CRSP(DFManipulation):
    """
    Useful functions to clean return data from CRSP \n
    """
    
    
    
# %%
# Seems useless


class ImportData():
    """
    **Unfinished.**
    Provides methods to \n
        - Specify data file path
        - Get a list of the column names for easy edits
        - Some summary statistics
    """

    def __init__(self, path='_MAC_', filename=''):
        """
        docstring
        """
        self.path = path
        self.filename = filename

    def import_data(file_type='.csv'):
        """
        Return a DataFrame from reading the file\n
        path + filename + file_type
        """
        pass
