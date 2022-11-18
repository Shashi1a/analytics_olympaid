from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pandas as pd
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import  Pipeline
from pipelines import  pipe_line
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import StackingClassifier

class UtilityFn:
    '''
    This class contains methods that will be used to split the data, 
    print the metric score for the training and validation data and 
    output the result to a submission file 
    '''

    ### Split the data into training and validation data
    @staticmethod
    def split_data(other: object):
        '''
        This function will split the data into traininig and validation data and labels
        ### Parameters: 
            other(object): instance of the class
        ### Return:
            None
        '''
        other.X = other.train[other.features]
        other.y = other.train[other.label]
        other.X_train, other.X_test, other.y_train, other.y_test = \
            train_test_split(other.X, other.y, test_size=0.2, random_state=42,
                             stratify=other.y)

    @staticmethod
    def print_score(ml_pipe, y_train, y_test, X_train: pd.DataFrame, X_test: pd.DataFrame):
        '''
        This function will simply output the score for the training and validation data 
        ### Paramaters: 
            ml_pipe(pipeline): machine learning pipe line
            y_train(array): training labels
            y_test(array): validation labels
            X_train(dataframe): data frame containing the training data
            X_test(dataframe): dataframe containing the validation data
        ### Return: 
            None
        '''
        tr_loss = log_loss(y_train, ml_pipe.predict_proba(X_train))
        val_loss = log_loss(y_test, ml_pipe.predict_proba(X_test))
        print(f'Log-loss for the training data: {tr_loss}')
        print(f'Log-loss for the validation data: {val_loss}')

    @staticmethod
    def data_submit(ml_pipe, df: pd.DataFrame, features: list, clfname: str):
        ''' 
        Function to make predictions on the test data
        ### Parameters: 
            ml_pipe(pipeline): machine learning pipeline
            df(dataframe): dataframe containing the test data
            features(list): list containing features used for fitting
            clfname(str): name of the classifier
        ### Return: 
            None
        '''
        preds = pd.Series(ml_pipe.predict_proba(
            df[features])[:, 1], name='OUTCOME')

        #preds = pd.Series(ml_pipe.predict_proba(
        #            df)[:, 1], name='OUTCOME')

        preds.to_csv(f'submission_{clfname}.csv', index=False)
        print('----- Predictions Saved -------')

   

    @staticmethod
    def stackedPred(clf1, clf2, obj: object, filename: str):
        '''
        This function will use  stacking to improve our predictions
        ### Parameter: 
            clf1(classifier1): classifier 1 
            clf2(classifier2): classifier 2
            data(object): instance of the type data  
            filename(str):name of the file  
        ### Return: 
            None
        '''
        estimators = [
            ('clf1', clf1),
            ('clf2', clf2)]

        clf = StackingClassifier(estimators=estimators,
                                 final_estimator=LogisticRegression(),
                                 stack_method='predict_proba'
                                 )

        stack_pipe = Pipeline([
            ('data', obj.data_pipe),
            ('clf', clf)
        ])

        stack_pipe.fit(obj.X, obj.y)
        UtilityFn.print_score(stack_pipe, obj.y_train,
                              obj.y_test, obj.X_train, obj.X_test)
        UtilityFn.data_submit(stack_pipe, obj.test, obj.features, filename)


    ## create the pipeline using the optimized classifier and train it on
    ## full data and make predictions on the test data
    @staticmethod
    def create_full_pipe(obj: object, clf, clfname: str):
        '''
        This function can be used to create the new pipeline that will implement
        a classifier passes as clf
        ### Parameters:
            object(object): instance of the data class
            clf(classifier): sklearn classifier 
            clf_name(str): name of the classifier
        ### Return:  
            None
        '''
        pipe_line.ml_pipe(obj, clf)
        obj.ml_pipe.fit(obj.X, obj.y)

        UtilityFn.print_score(obj.ml_pipe, obj.y_train,
                            obj.y_test, obj.X_train, obj.X_test)
        UtilityFn.data_submit(obj.ml_pipe, obj.test, obj.features, clfname)
