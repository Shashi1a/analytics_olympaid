import  pandas as pd 


### Blue print to construct new features using older features
class FeatureEngineering:
    '''
    This will create new features based on other features 
    '''
    @staticmethod
    def convert_cat(df: pd.DataFrame, colname: str, tolr: list, catname: str):
        '''
        This function will convert the features with integer values into 
        categories based on list tolr
        ### Parameter:
        df(dataframe): dataframe containing the data
        colname(str): name of the column with integer values
        tolr(list): list containing the values that we want to convert to categories
        catname(list): category value that we want to give 
        ### Return: 
        '''
        ids = df[colname].isin(tolr)
        df.loc[ids, f'{colname}_CAT'] = catname
        return df

    @staticmethod
    def convertDUIS(other: object):
        '''
        This function will convert duis features into categorical features
        by clubbing certain values in the given range
        ### Parameters: 
            other(object): instance of a class
        ### Return: 
            None 
        '''

        other.train = FeatureEngineering.convert_cat(
            other.train, 'DUIS', [0], 'no')
        other.train = FeatureEngineering.convert_cat(
            other.train, 'DUIS', [1, 2, 3], 'low')
        other.train = FeatureEngineering.convert_cat(
            other.train, 'DUIS', [4, 5, 6], 'high')

        other.test = FeatureEngineering.convert_cat(
            other.test, 'DUIS', [0], 'no')
        other.test = FeatureEngineering.convert_cat(
            other.test, 'DUIS', [1, 2, 3], 'low')
        other.test = FeatureEngineering.convert_cat(
            other.test, 'DUIS', [4, 5, 6], 'high')

    @staticmethod
    def convertSpeed(other: object):
        '''
        This function will convert the speed violations feature into a categorical
        Feature by binning values within the range
        ### Parameters:
            other(object): instance of a class 
        ### Return:  
            None
        '''
        other.train = FeatureEngineering.convert_cat(
            other.train, 'SPEEDING_VIOLATIONS', [0], 'no')
        other.train = FeatureEngineering.convert_cat(
            other.train, 'SPEEDING_VIOLATIONS', [i for i in range(1, 6)], 'low')
        other.train = FeatureEngineering.convert_cat(
            other.train, 'SPEEDING_VIOLATIONS', [i for i in range(6, 11)], 'medium')
        other.train = FeatureEngineering.convert_cat(
            other.train, 'SPEEDING_VIOLATIONS', [i for i in range(11, 21)], 'high')

        other.test = FeatureEngineering.convert_cat(
            other.test, 'SPEEDING_VIOLATIONS', [0], 'no')
        other.test = FeatureEngineering.convert_cat(
            other.test, 'SPEEDING_VIOLATIONS', [i for i in range(1, 6)], 'low')
        other.test = FeatureEngineering.convert_cat(
            other.test, 'SPEEDING_VIOLATIONS', [i for i in range(6, 11)], 'medium')
        other.test = FeatureEngineering.convert_cat(
            other.test, 'SPEEDING_VIOLATIONS', [i for i in range(11, 21)], 'high')

    @staticmethod
    def convertAccident(other: object):
        ''' 
        This function will convert PAST_ACCIDENTS column into a categorical
        feature by binning values within a range
        ### Parameter: 
            other(object): instance of a class
        ### Return: 
            None
        '''
        other.train = FeatureEngineering.convert_cat(
            other.train, 'PAST_ACCIDENTS', [0], 'no')
        other.train = FeatureEngineering.convert_cat(
            other.train, 'PAST_ACCIDENTS', [i for i in range(1, 6)], 'low')
        other.train = FeatureEngineering.convert_cat(
            other.train, 'PAST_ACCIDENTS', [i for i in range(6, 11)], 'medium')
        other.train = FeatureEngineering.convert_cat(
            other.train, 'PAST_ACCIDENTS', [i for i in range(11, 16)], 'high')

        other.test = FeatureEngineering.convert_cat(
            other.test, 'PAST_ACCIDENTS', [0], 'no')
        other.test = FeatureEngineering.convert_cat(
            other.test, 'PAST_ACCIDENTS', [i for i in range(1, 6)], 'low')
        other.test = FeatureEngineering.convert_cat(
            other.test, 'PAST_ACCIDENTS', [i for i in range(6, 11)], 'medium')
        other.test = FeatureEngineering.convert_cat(
            other.test, 'PAST_ACCIDENTS', [i for i in range(11, 16)], 'high')

    @staticmethod
    def createFeat(df: pd.DataFrame):
        '''
            This function is called when one wants to combine married and child features into one
            ### Parameter: 
                df(pd.DataFrame): dataframe containing the data
            ### Return: 
                df(pd.DataFrame): dataframe containing the data 
            '''

        id1 = df[(df.MARRIED == 0) & (df.CHILDREN == 0)].index
        id2 = df[(df.MARRIED == 0) & (df.CHILDREN == 1)].index
        id3 = df[(df.MARRIED == 1) & (df.CHILDREN == 0)].index
        id4 = df[(df.MARRIED == 1) & (df.CHILDREN == 1)].index

        df.loc[id1, 'MAR_CHILD'] = 'NO_NO'
        df.loc[id2, 'MAR_CHILD'] = 'NO_YES'
        df.loc[id3, 'MAR_CHILD'] = 'YES_NO'
        df.loc[id4, 'MAR_CHILD'] = 'YES_YES'
        return df

    @staticmethod
    def marriedChild(other: object):
        '''
        This function will convert married and child two features into one categorical features
        ### Parameters:
            other(object): instance of the class Data 
        ### Return: 
            None 
        '''
        other.train = FeatureEngineering.createFeat(other.train)
        other.test = FeatureEngineering.createFeat(other.test)
