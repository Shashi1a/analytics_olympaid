### DataPlot class to perform visualization for the data
import seaborn as sns 
import matplotlib.pyplot as plt
import  numpy as np 
class DataPlot:
    ''' 
    This class inherits from the DataPrep class and main functionality is to 
    plot the data
    ''' 

    @staticmethod
    def freqPlot(other: object, col: str, colname1: str):
        '''
        This function will create the fequency plot using the value_counts utility from the pandas
        and training and the test data
        ### Parameters:
            other(object): instance of the other class
            col(str): name of the features
            colname1(str): x-axis label
        ### Return:  
            None 
        '''
        xax = other.train[col].value_counts().index
        yax = other.train[col].value_counts().values

        xax2 = other.test[col].value_counts().index
        yax2 = other.test[col].value_counts().values

        f, ax = plt.subplots(1, 2, figsize=(13, 4))
        g = sns.barplot(x=xax, y=yax, ax=ax[0])
        g.set_xlabel(f'{colname1}', fontsize=16)
        g.set_ylabel('FREQUENCY', fontsize=16)
        g.set_title('Training Data', fontsize=18)

        g1 = sns.barplot(x=xax2, y=yax2, ax=ax[1])
        g1.set_xlabel(f'{colname1}', fontsize=16)
        g1.set_ylabel('FREQUENCY', fontsize=16)

        g1.set_title('Testing Data', fontsize=18)
        plt.show()

    @staticmethod
    def histPlot(other:object, col: str, colname: str):
        '''
        This function will plot the histogram of the numerical features 
        ### Parameters:
            other(object): instance of the class
            col(str): feature we want to plot
            colname(str): name of the feature 
        ### Return:
            None
        '''

        f, ax = plt.subplots(1, 2, figsize=(13, 4))
        g = sns.histplot(data=other.train, x=col, ax=ax[0])
        g.set_xlabel(f'{colname}', fontsize=16)
        g.set_ylabel(f'COUNT', fontsize=16)
        g.set_title('Training Data', fontsize=18)

        g1 = sns.histplot(data=other.test, x=col, ax=ax[1])
        g1.set_xlabel(f'{colname}', fontsize=16)
        g1.set_ylabel(f'COUNT', fontsize=16)
        g1.set_title('Training Data', fontsize=18)
        plt.show()

    @staticmethod
    def plotIDs(obj: object, nvals: int, col: str, xlab: str, ylab: str):
        '''
        THis function will plot the ID columns (POSTAT_CODE and ID) given the object
        ### Parameter:
            obj(object): instance of the data class
            nvals(int): number of most frequent values to use
            col(str): name of the feature
            xlab(str): x-axis label 
            ylab(str): y-axis label 
        ### Return:
            None
        '''
        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        df1 = obj.train[col].value_counts().nlargest(nvals)

        g0 = sns.barplot(x=df1.index, y=np.log10(
            df1.values), order=df1.index, ax=ax)
        ax.tick_params(axis='x', labelrotation=90)
        g0.set_xlabel(f'{xlab}', fontsize=18)
        g0.set_ylabel(f'{ylab}', fontsize=18)
        g0.set_title(f'{nvals} Frequent values',fontsize=20)
        plt.show()

    