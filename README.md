# Insurance claim prediction

**`analysis.ipynb`** is the main working notebook.  It calls various functions stored in other files to perform the prediction task. Link for a working implementation of the code using [**_analysis.ipynb_**](https://shashi1a.github.io/analytics_olympaid/)

I will explain the working of the notebook  very briefly but before that I will explain the `Data Cleaning` and `Preprocessing` steps

## Data Cleaning and Preprocessing

I will discuss briefly the main data-cleaning steps. Although the data we have looks clean. However, in production line one can receive the data that is not clean for that scenario we have to incorporate methods that can help in the modelling task.

* We need to deal with missing value. This comes under imputation

  1. **Imputation**: We can have missing values due to human-error or absence of data. This can occur in either numeric feature or categorical features. Here I menion two very simple stratergy to perform imputation one for numeric feature and one for categorical feature.
     1. `most-frequent` for categorical features.
     2. `median` for numeric features.
     3. ```python
        ### to perform imputation for categorical feature
        cat_imp = SimpleImputer(statergy='most-frequent')

        ### to perform imputation for numerical feature
        num_imp = SimpleImputer(statergy='median')

        ```
  2. **Data Preprocessing**: Some machine learning algorithm expects numeric feature values to be in similar scale so we need to perform transformation in our data such that they are in same scale. Also all machine learning models can't handle categorical features so we need to transform them as well in a form that a machine learning algorithm can use.
     1. `StandardScaler()` for numeric features.
     2. `One-Hot encoding` and `Ordinal encoding` for categorical features.
     3. **`Tree`** based models are okay with `Ordinal encoding` but  **`Linear models and neural network`**     works best with`One-Hot encoding
     4. ````python
        ### perform scaling for numeric feature
        num_scal = StandardScaler().fit_transform(num_feat)

        ### perform one-hot encoding for a categorical feature
        cat_oh = OneHotEncoder().fit_transform(cat_feat,handle_unknown='ignore')

        ### perform ordinal encoding
        cat_oe = OrdinalEncoder().fit_transform(cat_feat)
        ```
        ````

### **`load_data.py`**

It contains the class `DataCreate`. This class serves multi-purpose. If has `train`, `test`,`features`and`label` as attribute and has following functions.

1. `load_data()`: To load test data and train data files.
2. `labelandFeatures()`: to set the feature and label names
3. `convertDatatype()`: To convert datatype of features if features doesn't have appropriate datatype.

   ```python
   ### create the data object
   data = DataCreate('train.csv': str, 'test.csv':str)

   ### load the data
   data.load_data()

   ### set label and features
   label = 'OUTCOME'
   data.labelandFeatures(label)

   ### Convert data type of features
   data.convertDatatype()

   ```

### **`dataplot.py`**

Contains the `DataPlot` class that has  functions I used to plot the categorical and numeric features for both the training and test data.

1. `FreqPlot()`: Function to frequency plot of a **categorical** feature present in training and test data
2. `histPlot()`: Plot histogram of **numeric** features
3. `plotIDs()`: Function to plot **ID** like features. Plots only top categories supplied by user.

```python
### plot a categorical feature cat_1 
DataPlot.FreqPlot(data.train: pd.DataFrame, data.test: pd.DataFrame,cat_1: str, cat_1name: str)

### plot a numeric feature
DataPlot.histPlot(data.train: pd.DataFrame, data.test: pd.DataFrame, 'col': str, 'colname':str)

### plot Id feature
DataPlot.plotIDs()
```

### **`pipelines.py`**

Contains the class `pipe_line` that has function to create various data pipelines and machine learning pipeline.

1. `data_pipeline_tree()`: Function to create data pipeline for tree based model.
2. `data_pipeline_linear()`: Function to create data pipeline for linear models.
3. `ml_pipe()`: To create a machine learning pipeline. It employs a data pipeline and a classifier.

### **`idfeats.py`**

Contains a class `IDfeatures` that has functions to perform preprocessing of ID features.

1. **`postal_code_features()`**: Functions to use postal_code features. It keeps top `nvals` feature values in separate class and club rest of them in one category. Can be useful for high cardinality categorical feature
   s where category distribution is skewed.
2. `ID_features()`: Function to use ID features. Keeps only top `nvals` feature values and put rest of the rare values in one category. Similar to the `postal_code_features()`.

### **`utilitfn.py`**

Contains `UtilityFn` class that contains few commonly used functions.

1. **`spit_data()`**: To separate the training data into features (**X**) and labels (**y**). Also split the data into training data and validation data that keeps 20% of the data for the validation. The data is split into a stratified way based on the label  **y**.
2. **`print_score()`**: This functions takes as an argument the trained machine learning pipeline (ml_pipe) and computes the log-loss for the training and validation data.
3. **`data_submit()`**: This function creates the submission file that stored prediction probability for the test set. It takes trained ml pipeline as an argument, features (to be used for the prediction) and the submission file name.
4. **`stackedPred()`**: Can be used to stack two pretrained classifiers to improve the predictions made by individual models.
5. **`create_full_pipe()`**: This function creates the full machine learning pipeline, fit the model on the training data, prints the performance of the model and stores the submission file for the test data.

### **`tunemodel.py`**

This file contains the `ParamTune class`. It contains functions to tune the hyperparameter of a *random-forest model* and a *gradient-boosting classifier model*.

1. **`tune_params_gb()`**: Performs hyperparameter tuning for a gradient-boosting classifier on the training and validation data and the ml pipeline using the `optuna` library. The function returns the optimized values of the model parameter and stores them. They can be used later for training and making predictions on the test data.
2. **`tune_params_rf()`**: Used to optimize the random forest classifier. The metric that is optimized is log-loss and it returns the hyper-parameter of the optimized model.

### **`featEng_1.py`**

Class that contain methods to create new features.
