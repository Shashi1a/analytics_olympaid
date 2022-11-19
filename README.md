# Insurance claim prediction

This will predict the 

**`analysis.ipynb`** is the main working notebook. It calls various functions declared in other files.

1. **`load_data.py`** contains the class DataCreate. This class serves multi-purpose. If has `train`, `test`,`features` and `label` as attribute and has following functions.

   1. `load_data()`: To load test data and train data files.
   2. `labelandFeatures()`: to set the feature and label names
   3. `convertDatatype()`: To convert datatype of features if features doesn't have appropriate datatype.
2. **`dataplot.py`** contains the DataPlot class that has  functions I used to plot the categorical and numeric features for both the training and test data.

   1. `FreqPlot()`: Function to frequency plot of a **categorical** feature present in training and test data
   2. `histPlot()`: Plot histogram of **numeric** features
   3. `plotIDs()`: Function to plot **ID** like features. Plots only top categories supplied by user.
3. **`pipelines.py`** contains the class `pipe_line` that has function to create various data pipelines.

   1. `data_pipeline_tree()`: Function to create data pipeline for tree based model.
   2. `data_pipeline_linear()`: Function to create data pipeline for linear models.
   3. `ml_pipe()`: To create a machine learning pipeline. It employs a data pipeline and a classifier.
   4. **Imputation**: To deal with missing values.

      1. `most-frequent` for categorical features
      2. `median` for numeric features
   5. **Data Preprocessing**: To transform values such that algorithm can use them.

      1. `StandardScaler()` for numeric features
      2. `One-Hot encoding` and `Ordinal encoding` for categorical features.
      3. **`Tree`** bases models are okay with `Ordinal encoding` but  **`Linear models and neural network`**works best with`One-Hot encoding`
   6. **`idfeats.py`**: Contains a class IDfeatures that has functions to perform preprocessing of ID features.

      1. **`postal_code_features()`**: Functions to use postal_code features. It keeps top `nvals` feature values in separate class and club rest of them in one category. Can be useful for high cardinality categorical features where category distribution is skewed.
      2. `ID_features()`: Function to use ID features. Keeps only top `nvals` feature values and put rest of the rare values in one category. Similar to the `postal_code_features()`.
   7. **`UtilityFn`**
   8. **`ParamTune.py`**
   9. **`FeatureEngineering.py`**
   10.
