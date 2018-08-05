
# coding: utf-8

# # Database - Pima Indians Diabetes
# 
# Motivation - Help medical professionals to make diagnosis easier by bridging gap between huge datasets and human knowledge.
# 
# Apply machine learning techniques for given classification in a dataset that describes a population that is under a high risk of the onset of diabetes.

# In[15]:

# Importing Data Analysis Toolkit
import pandas as pd
import numpy as np


# Importing Data Viz toolkit
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')


get_ipython().magic("config InlineBackend.figure_format = 'retina'  # Setting the right figure format")
import warnings
warnings.filterwarnings('ignore')


# In[16]:

pima = pd.read_csv("diabetes.csv")  # file should be in the same directory of Jupyter Notebook


# In[17]:

pima.head()  # To just have a look at the data


# # Additional details about the attributes
# 
# Pregnancies:     Number of times pregnant
# 
# Glucose:         Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 
# BloodPressure:   Diastolic blood pressure (mm Hg)
# 
# SkinThickness:   Triceps skin fold thickness (mm)
# 
# Insulin:         2-Hour serum insulin (mu U/ml)
# 
# BMI:             Body mass index (weight in kg/(height in m)^2)
# 
# DiabetesPedigreeFunction:    Diabetes pedigree function
# 
# Age:            Age (years)
# 
# Outcome:        Class variable (0 or 1)

# In[18]:

pima.shape  # To check rows, columns (Data points, Parameters)


# In[19]:

pima.describe()  # Basic Stat description


# In[20]:

pima.groupby("Outcome").size()  # Grouping by the last Parameter to check the number of Positives and Negatives


# In[21]:

a = 500/768


# In[22]:

a


# In[23]:

b = 1 - a


# In[24]:

b


# In[25]:

# So we see that 65.10% are negatives and 34.89% are positives.


# In[26]:

pima['Outcome'].value_counts().plot('bar')  # Just seeing the same stuff on bars


# In[27]:

pima.hist(figsize=(12,12))   # Visualizing each parameter


# Attributes BMI, BloodPressure, Glucose are found to be normally distributed. 
# 
# BMI and BloodPressure nearly have Gaussian distribution. 
# 
# Age, DiabetesPedigreeFunction, Insulin, Pregnancies found to be exponentially distributed.

# In[28]:

# X axis is the data range. Y axis is How many data points have that data.


# In[29]:

pima.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(12,12)) 

# Box plot helps to know the median value, Inter Quartile Range and Overall distribution of the data.


# Observed that spread of attributes is quite different.
# 
# Attributes Age, Insulin appear to be quite skewed towards smaller values. 
# 
# 
# Scaling on dataset can be applied during data pre-processing.

# In[30]:

sns.heatmap(pima.corr(), annot=True)

# pima.corr() checks for the cross tab correlation between each paramter and 

# we are visualizing it on a heat map. Just seeing one half of the map will suffice.


# Observed that attributes BloodPressure, SkinThickness are not much related to outcome. 
# 
# Feature extraction can be tried to observe performance.

# 
# 
# # Data Pre-processing

# Note : Replaced 0 values by mean, but no performance improvement was observed while evaluating models. 
# Dropped rows with 0 values, performance seems to be improved. But dataset reduces to half. 
# Hence commented below lines.

# In[31]:

# Data preprocessing - replace zeroes with mean or drop records with 0 values.
# attributes_to_replace_zero =list(pima.columns[1:6])      # list all column names. 
# pima[attributes_to_replace_zero] = pima[attributes_to_replace_zero].replace(0, np.NaN)
# pima.fillna(dataset.mean(), inplace=True) 
# pima.dropna(inplace=True)


# In[32]:

# Split into Input and Output.
attributes = list(pima.columns[:8])  # creates a list of all paramter names
X = pima[attributes].values  # masking the parameter values
y= pima['Outcome'].values  # Just picking up values from Outcome.


# In[33]:

attributes


# In[34]:

X


# In[35]:

y


# In[36]:

# Now scaling our input data


# In[37]:

from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() 

X = sc_X.fit_transform(X)  # Transforming


# In[38]:

X  # Data after scaling and transforming


# Note : Normalization reduced performance while evaluating models. Hence code disabled.

# In[39]:

# from sklearn import preprocessing
# X = preprocessing.normalize(X)


# Split into train and test sets.

# In[40]:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state =0)


# In[41]:

X_train


# In[42]:

X_test


# In[43]:

y_train


# In[44]:

y_test


# Applied feature selection, but not much change in performance. So code lines disabled.

# In[45]:

#Applying Kernel PCA (Principle Component Analysis) ( Not much change in performance) 
#from sklearn.decomposition import PCA
#pca = PCA(n_components = 6) 

#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)

#explained_variance = pca.explained_variance_ratio_

# It came out to be low


# explained_variance  array([ 0.89142243,  0.059357  ,  0.02545099,  0.01317226,  0.00716861,
#         0.00290131])

# # Evaluating Models

# In[46]:

# Importing the entire classifier suite of algorithms.

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV


# In[48]:

# Creating objects of required models.

# Within a list storing all the objects in a tuple form -> (nameofalgo, objectofalgo)
models = []
models.append(("LR",LogisticRegression()))
models.append(("GNB",GaussianNB()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("DecisionTree",DecisionTreeClassifier()))
models.append(("LDA",  LinearDiscriminantAnalysis()))
models.append(("QDA",  QuadraticDiscriminantAnalysis()))
models.append(("AdaBoost", AdaBoostClassifier()))
models.append(("SVM Linear",SVC(kernel="linear")))
models.append(("SVM RBF",SVC(kernel="rbf")))
models.append(("Random Forest",  RandomForestClassifier()))
models.append(("Bagging",BaggingClassifier()))
models.append(("Calibrated",CalibratedClassifierCV()))
models.append(("GradientBoosting",GradientBoostingClassifier()))
models.append(("LinearSVC",LinearSVC()))
models.append(("Ridge",RidgeClassifier()))


# In[49]:

# Finding accuracy of models.

results = []
for name,model in models:
    kfold = KFold(n_splits=10, random_state=0)
    
# KFold will provide train/test indices to split data in train and test sets. 
# It will split dataset into k (here 10) consecutive folds (without shuffling by default).
# Each fold is then used a validation set once while the k - 1 remaining folds form the training set  


    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
# It gives you an unbiased estimate of the actual performance you will get at runtime
    
    results.append(tuple([name,cv_result.mean(), cv_result.std()]))
  
results.sort(key=lambda x: x[1], reverse = True)    
for i in range(len(results)):
    print('{:20s} {:2.2f} (+/-) {:2.2f} '.format(results[i][0] , results[i][1] * 100, results[i][2] * 100))


# # Optimize peformance of best model

# SVM Linear seems performs best. Now let us try to find the optimistic parameters for SVM using GridSearchCV.

# In Support Vector Machine (SVM), we need to choose different parameters to optimize our algorithms.
# 
# Choice of kernel (Similarity function)
# Linear kernel
# Polynomial kernel
# Logisitic/ Sigmoid kernel
# Gaussian/RBF kernel
# Choice of parameter C
# Choice of Gamma ( if using Gaussian kernel)
# Parameter C
# 
# The C parameter controls the tradeoff between classification of training points accurately and a smooth decision boundary or in a simple word, it suggests the model to choose data points as a support vector.

# The value of gamma and C should not be very high because it leads to the overfitting or it shouldn’t be very small (underfitting). Thus we need to choose the optimal value of C and Gamma in order to get a good fit.

# In[53]:

from sklearn.model_selection import GridSearchCV
model = SVC()
paramaters = [
             {'C' : [0.01, 0.1, 1, 10, 100, 1000], 'kernel' : ['linear']}   
    

    # We take C values as prescribed by data analysts.
    # We choose linear to keep it according to our data model.
             ]
grid_search = GridSearchCV(estimator = model, 
                           param_grid = paramaters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_ 
best_parameters = grid_search.best_params_  

print('Best accuracy : ', grid_search.best_score_)
print('Best parameters :', grid_search.best_params_  )


# # Finalize model

# In[54]:

# Predicting output for test set. 

final_model = SVC(C = 0.1, kernel = 'linear')
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score
cf = confusion_matrix(y_test, y_pred)
print(cf)
print(accuracy_score(y_test, y_pred) * 100) 


from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)


# Conclusion :- Observed accuracy of 82.46% on test set using SVM linear model.

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



