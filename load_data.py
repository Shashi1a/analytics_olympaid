### Data Create class for loading and manipulating the data
import pandas as pd 

class DataCreate:
    ''' 
    Class blue print for the DataPrep Class
    This function will load the data 
    '''

    def __init__(self: object,file1='train.csv',file2='test.csv'):
        '''
        Initializer for the class DataPrep
        ### Parameters:
            self(object): instance of the class
            file1(str): file containing training data
            file2(str): file containing testing data 
        ### Return:  
        '''
        self.file1 = file1 
        self.file2 = file2
    
    def load_data(self: object):
        '''
        This function will load the data from the csv files
        ### Parameters: 
            self(object): instance of the class
        ### Return: 
            None
        '''
        self.train = pd.read_csv(self.file1)
        self.test = pd.read_csv(self.file2)


    def labelandFeatures(self: object,label:str):
        '''
        This function will name the label and features in the dataset.
        ### Parameters: 
            self(object): instance of the class 
            label(str): name of the label column
        ### Return: 
            None 
        '''
        self.label=label
        self.features=self.train.columns[self.train.columns!=self.label]
        #self.convertDatatype()
        

    ### To chenge the datatype of columns to correct values
    def convertDatatype(self:object):
        '''
        This function will convert the datatype of the given features. 
        ### Parameters:
            self(object): instance of the class
        ### Return:
            None
        '''
        ## Changing id feature datatype for both training and test data
        self.train['ID'] = self.train.ID.astype('object')
        self.test['ID'] = self.test.ID.astype('object')

        ### changing float datatype to category for training data
        self.train['VEHICLE_OWNERSHIP'] = self.train.VEHICLE_OWNERSHIP.astype('object')
        self.train['MARRIED'] = self.train.MARRIED.astype('object')
        self.train['CHILDREN'] = self.train.CHILDREN.astype('object')
        self.train['ANNUAL_MILEAGE'] = self.train.ANNUAL_MILEAGE.astype('object')

        ### changing float datatype to category for test data
        self.test['VEHICLE_OWNERSHIP'] = self.test.VEHICLE_OWNERSHIP.astype('object')
        self.test['MARRIED'] = self.test.MARRIED.astype('object')
        self.test['CHILDREN'] = self.test.CHILDREN.astype('object')
        self.test['ANNUAL_MILEAGE'] = self.test.ANNUAL_MILEAGE.astype('object')

        ### changing int datatype to category for training data
        self.train['POSTAL_CODE'] = self.train.POSTAL_CODE.astype('object')
        self.train['SPEEDING_VIOLATIONS'] = self.train.SPEEDING_VIOLATIONS.astype('object')
        self.train['DUIS'] = self.train.DUIS.astype('object')
        self.train['PAST_ACCIDENTS'] = self.train.PAST_ACCIDENTS.astype('object')

        ### changing int datatype to category for testing data
        self.test['POSTAL_CODE'] = self.test.POSTAL_CODE.astype('object')
        self.test['SPEEDING_VIOLATIONS'] = self.test.SPEEDING_VIOLATIONS.astype('object')
        self.test['DUIS'] = self.test.DUIS.astype('object')
        self.test['PAST_ACCIDENTS'] = self.test.PAST_ACCIDENTS.astype('object')
        