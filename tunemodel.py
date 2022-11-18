import  optuna  
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.ensemble import  RandomForestClassifier
from pipelines import  pipe_line
from sklearn.metrics import log_loss

class ParamTune:
    '''
    This class contains method to perform hyper parameter tuning for the random
    forest and gradient boosting method 

    '''
    @staticmethod
    def tune_params_gb(obj: object, ntrials: int):
        '''
        This function will perform hyper parameter testing for gradient boosted trees 
        ### Parameters: 
            obj(object): instance of the datafeat class
        ### Return:
            param_dict(dictionary): dictionary with tuned hyper parameters.
        '''

        def objective(trial):
            ''' 
            Objective function that will be used to tune the hyperparameter of the model
            '''

            max_depth = trial.suggest_int('max_depth', 10, 25)
            n_estimators = trial.suggest_int('n_estimators', 90, 140)
            learning_rate = trial.suggest_float('learning_rate', 0.003, 0.1)
            min_samples_split = trial.suggest_float(
                'min_samples_split', 0, 1.0)
            #min_samples_leaf = trial.suggest_float('min_samples_leaf',0,0.5)

            clf = GradientBoostingClassifier(max_depth=max_depth,
                                             n_estimators=n_estimators,
                                             #min_samples_leaf=min_samples_leaf,
                                             min_samples_split=min_samples_split,
                                             learning_rate=learning_rate, random_state=42)

            pipe_line.ml_pipe(obj, clf)
            obj.ml_pipe.fit(obj.X_train, obj.y_train)
            return log_loss(obj.y_test, obj.ml_pipe.predict_proba(obj.X_test))

        study = optuna.create_study()
        study.optimize(objective, n_trials=ntrials)
        return study.best_params

    @staticmethod
    def tune_params_rf(obj, ntrials: int):
        '''
        This function will perform hyper parameter testing for gradient boosted trees 
        ### Parameters: 
            data(object): instance of the datafeat class
            ntrials(int): number of trials for the hyper parameter tuning
        ### Return:
            param_dict(dictionary): dictionary with tuned hyper parameters.
        '''

        def objective(trial):
            ''' 
            Objective function that will be used to tune the hyperparameter of the Random Forest model
            '''

            max_depth = trial.suggest_int('max_depth', 15, 25, 1)
            n_estimators = trial.suggest_int('n_estimators', 50, 150)
            min_samples_split = trial.suggest_int('min_samples_split', 10, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 10, 17)
            criterion = trial.suggest_categorical('criterion',['gini','entropy'])
            clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,
                                         min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf, random_state=42,
                                         criterion=criterion)

            pipe_line.ml_pipe(obj, clf)
            obj.ml_pipe.fit(obj.X_train, obj.y_train)
            return log_loss(obj.y_test, obj.ml_pipe.predict_proba(obj.X_test))

        study = optuna.create_study()
        study.optimize(objective, n_trials=ntrials)
        return study.best_params
