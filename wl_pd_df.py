# -> Created on 26 November 2020
# -> Author: Weiguang Liu
# %%
import numpy as np 
import  pandas as pd
from matplotlib import pyplot as plt
# %%


class DFManipulation():
    """
    Provides methods to manipulate a Pandas Dataframe. \n
    See the wiki page for list of methods.
    """
    def __init__(self, *DF):
        if DF == ():
            self.DF = self.gen_test_df()
            print(f"Warning: No dataframe given, using a generate dataframe for testing\n{self.DF}")
        else:
            self.DF = DF[0]

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
                break
            elif i + window_size > T-1:
                starts = starts + [date_list[i]]
                ends = ends + [date_list[-1]]
            else:
                starts = starts + [date_list[i]]
                ends = ends + [date_list[i + window_size]]
        num_of_windows = len(starts)
        DF_rolling = [DF.loc[(DF.index >= starts[i]) * (DF.index < ends[i])]
                    for i in range(num_of_windows)]
        return DF_rolling
    
    def show_number_of_nan(self, prune = False):
        """
        This method shows the number of NaN in the columns over the index. 
        """
        DF= self.DF
        n_nan = DF.isna().sum(axis=1)
        idx_all_col_are_nan = n_nan[n_nan == len(DF.columns)].index
        if prune:
            DF = DF.drop(idx_all_col_are_nan)
        elif len(idx_all_col_are_nan) > 0:
            print(f'For the following dates, all columns are NaN. To delete them, run show_number_of_nan(prune == True):\n{n_nan[n_nan==len(M.DF.columns)]}')
        plt.plot(n_nan)
        return n_nan
        
    def gen_test_df(self):
        """
        Generate a test DataFrame
        """
        dd = {'a' : [1,2,3,4,5], 'b' : [0,0,0,0,0], 'c': [np.nan,np.nan,np.nan,np.nan,np.nan],'d' : [np.nan,np.nan, 3,2,1]}
        df = pd.DataFrame(dd)
        return df
# %%
# Specialized to CRSP

class CRSP(DFManipulation):
    """
    Methods specialized to CRSP data handling
    """
    def __init__(self, *DF):
        super().__init__(*DF)
        self.FACTOR = self.get_factor()
    
    def clean_return(self, date_column_name='date', original_format='%Y%m%d', return_column_name='RET'):
        """
        This provides a common handling of CRSP return data
        """
        self.convert_datetime(None, date_column_name, original_format)
        self.rid_of_BC(return_column_name)
        return self.DF

    def convert_datetime(self,DF = None, date_column_name='date', original_format='%Y%m%d', **kwargs):
        """
        Convert the date column to datetime index
        """
        if DF == None:
            DF = self.DF
            save = True
        else:
            save = False
        # print(DF)
        if date_column_name in DF.columns:
            DF[date_column_name] = pd.to_datetime(DF[date_column_name], format= original_format)
            print(f"We have converted the column {date_column_name}, the date range is\nStart date: {DF[date_column_name].iloc[0]}\nEnd date:{DF[date_column_name].iloc[-1]}\n")
        else:
            print(f'There is no columns called {date_column_name}, check the following column names\n{DF.columns.values}' )
        if save:
            self.DF = DF 
        else:
            return DF

    def rid_of_BC(self, return_column_name = 'RET'):
        """
        The method to get rid of 'B' and 'C' in CRSP RET. 
        """
        DF = self.DF
        if return_column_name in DF.columns:
            select_condition = (DF[return_column_name] == 'C') | (  DF[return_column_name] == 'B')
            DF = DF.drop(DF[select_condition].index)
            DF.return_column_name = DF[return_column_name].astype('float')
            print(f'We have deleted the entries with values \'B\',\'C\' in the colume {return_column_name} and converted the entries to float type')
            self.DF = DF
        else:
            print(f'There is no columns called {return_column_name}, check the following column names\n{DF.columns.values}' )

    def get_factor(self):
        FACTOR = pd.read_csv(
            '/Users/lwg342/OneDrive - University of Cambridge/Projects/Covariance Estimation with ML/Data/CRSP_4Factor_Return_2006-2019.csv', index_col='date')
        FACTOR.index = pd.to_datetime(FACTOR.index, format='%Y%m%d')
        return FACTOR
# %%
