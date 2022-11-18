from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class pipe_line:
    '''
    This class defines the function requires to  create the pipeline required 
    for the tree method  using the dataframe and features 
    '''
    @staticmethod
    def data_pipeline_ord(other: object):
        '''
        This function will return the data pipeline given features datatype
        ### Parameters: 
            other(object): instance of the class
        ### Return: 
            None
        '''

        pipe_num = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('transform', StandardScaler())])

        pipe_cat = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('transform', OrdinalEncoder())
        ])

        pipe_data = ColumnTransformer([
            ('ord_col', pipe_cat, other.cat_col),
            ('num_col', pipe_num, other.num_col)
        ])
        other.data_pipe = pipe_data

    @staticmethod
    def data_pipeline_ohot(other: object):
        '''
        This function will return the data pipeline given features datatype
        ### Parameters: 
            other(object): instance of the class
        ### Return: 
            None
        '''
        pipe_cat = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('transform', OneHotEncoder(handle_unknown='ignore'))])

        pipe_num = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('transform', StandardScaler())])

        
        pipe_data = ColumnTransformer([
            ('oh_col', pipe_cat, other.cat_col),
            ('num_col', pipe_num, other.num_col)
        ])
        other.data_pipe = pipe_data


    @staticmethod
    def ml_pipe(other: object, clf):
        '''
        This function will create the entire ml pipeline using the data pipeline and 
        ml model
        ### Parameters:
            other(object): instance of the class
            clf: classifier instance 
        ### Return: 
            None
        '''
        data_pipe = other.data_pipe
        ml_pipe = Pipeline([
            ('data_pipe', data_pipe),
            ('ml', clf)
        ])
        other.ml_pipe = ml_pipe
