import  pandas as pd
import  matplotlib.pyplot as plt
import  seaborn as sns

class FeaturesBool:
    '''
    This class contains methods to convert features into boolean features based on threshold values 
    '''
    @staticmethod
    def plotFeaturetolr(df: pd.DataFrame, colname: str, tolr: int):
        '''
        This function will plot the new features that we have generated based on 
        the threshold 
        ### Parameters:
            df1(pd.DataFrame): data greater than threshold
            df2(pd.DataFrame): data less than the threshold
            colname(str): name of the column
            tolr(int): value threshold
        ### Return: 
            None
        '''
        f, ax = plt.subplots(1, 2, figsize=(13, 4))
        a0 = sns.countplot(data=df[df[colname] > tolr],
                           x=colname, hue='OUTCOME', ax=ax[0])
        a0.set_ylabel('Frequency', fontsize=14)
        a0.set_title(f'{colname} > {tolr}', fontsize=16)
        a1 = sns.countplot(data=df[df[colname] < tolr],
                           x=colname, hue='OUTCOME', ax=ax[1])
        a1.set_ylabel('Frequency', fontsize=14)
        a1.set_title(f'{colname} < {tolr}', fontsize=16)
        plt.show()

    @staticmethod
    def boolFeatures(df: pd.DataFrame, col: str, tolr: int):
        '''
        This function will convert features into boolean features based on the 
        value of features name (col) and value (tol)
        ### Parameters: 
            df(pd.DataFrame): dataframe containing the data
            col(str): feature name
            tolr(int): value used for comparison
        ### Return: 
            df(dataframe): dataframe with new column
        '''
        df[f'BOOL_{col}_{tolr}'] = df[col] < tolr
        return df

    @staticmethod
    def NewFeatureBools(other: object, col: str, tolr: int):
        '''
        This function will create boolean features for the entire dataset.
        ### Parameters:
            other(object): instance of the class
            col(string): feature name
            tolr(int): value used for the comparison
        ### Return:  
            df(dataframe): dataframe with new features
        '''
        other.train = FeaturesBool.boolFeatures(other.train, col, tolr)
        other.test = FeaturesBool.boolFeatures(other.test, col, tolr)