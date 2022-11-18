import pandas as pd
class IDFeatures:
    
    @staticmethod
    def postal_code_features(obj:object, nvals: int):
        '''
        This function will reduce really large number of postal code values to 
        fewere by keeping 20 most appeared post codes and chage lesser appearing post
        codes into one category
        ### Parameters:
            obj(object): object of the class DataLoad
            nvals(int): number of postal codes to keep
        ### Return: 
        '''
        codes_postal = list(
            obj.train.POSTAL_CODE.value_counts().nlargest(nvals).index)

        ids1 = obj.train.POSTAL_CODE.isin(codes_postal)
        
        obj.train['POSTAL_CODE'] = obj.train.POSTAL_CODE.astype('str')
        obj.test['POSTAL_CODE'] = obj.test.POSTAL_CODE.astype('str')

        obj.train.loc[ids1, 'NEW_POSTAL_CODE'] = obj.train.loc[ids1, 'POSTAL_CODE']
        obj.train.loc[ids1 == False, 'NEW_POSTAL_CODE'] = 'unknown'

        ids2 = obj.test.POSTAL_CODE.isin(codes_postal)

        obj.test.loc[ids2, 'NEW_POSTAL_CODE'] = obj.test.loc[ids2, 'POSTAL_CODE']
        obj.test.loc[ids2 == False, 'NEW_POSTAL_CODE'] = 'unknown'

        return obj

    @staticmethod
    def ID_features(df1: pd.DataFrame, df2: pd.DataFrame, nvals: int):
        '''
        This function will create categorical features from the ID columns.
        Since the number of unique IDs are very large one can't convert them into
        categorical features. We only consider top 10 
        ### Parameters: 
            df1(dataframe): dataframe containing training data
            df2(dataframe): dataframe containing test data
            nvals(int): top nvals used for encoding
        ### Return:
            df1(dataframe): training data with new ID column
            df2(dataframe): testing data with new ID columns
        '''

        id_vals = list(df1.ID.value_counts().nlargest(nvals).index)
        ids_tr = df1.ID.isin(id_vals)
        df1['ID'] = df1.ID.astype('str')
        df2['ID'] = df2.ID.astype('str')

        df1.loc[ids_tr, 'NEW_ID'] = df1.loc[ids_tr, 'ID']
        df1.loc[ids_tr == False, 'NEW_ID'] = 'unknown'

        ids_val = df2.ID.isin(id_vals)
        df2.loc[ids_val, 'NEW_ID'] = df2.loc[ids_val, 'ID']
        df2.loc[ids_val == False, 'NEW_ID'] = 'unknown'
        return df1, df2
