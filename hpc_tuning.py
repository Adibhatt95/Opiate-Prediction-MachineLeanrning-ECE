
# coding: utf-8

# # Foundations of Data Science - Final Project
# ## A Predictive Modeling and Analysis on Causes of Death by Opioid Prescription
# ### Submission By:
# 
# <ul>
#     <li>Aditya Bhatt (apb462@nyu.edu)</li>
# </ul>

# In[1]:


import pandas as pd
import sklearn
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

# Data Loading
# Load Drug Prescriber Data in Pandas Dataframe
drug_prescriber_df = pd.read_csv("prescriber_information.csv")

# Load Overdose Deaths Data in Pandas Dataframe
overdose_df = pd.read_csv("overdose_deaths.csv")

# Load Overdose Deaths Data in Pandas Dataframe
opioid_df = pd.read_csv("opioid_drug_list.csv")


# # Data Preprocessing

# In[2]:


# Data Preprocessing
import numpy as np
# Drug Prescriber Dataset Cleaning Credentials Column - Replace Missing Values with String Value
drug_prescriber_df.Credentials = drug_prescriber_df.Credentials.fillna('UNKNOWN')

# Clean Up States from Prescriber Data to match the list of states in Overdose Death Data
drug_prescriber_df= drug_prescriber_df[drug_prescriber_df.State != 'AE']
drug_prescriber_df = drug_prescriber_df[drug_prescriber_df.State != 'ZZ']
drug_prescriber_df = drug_prescriber_df[drug_prescriber_df.State != 'AA']
drug_prescriber_df = drug_prescriber_df[drug_prescriber_df.State != 'PR']
drug_prescriber_df = drug_prescriber_df[drug_prescriber_df.State != 'GU']
drug_prescriber_df = drug_prescriber_df[drug_prescriber_df.State != 'VI']

# Overdoses Data - Removing Commas in Numerical Values
overdose_df['Deaths'] = overdose_df['Deaths'].str.replace(',', '')
overdose_df['Deaths'] = overdose_df['Deaths'].astype(int)
overdose_df['Population'] = overdose_df['Population'].str.replace(',', '')
overdose_df['Population'] = overdose_df['Population'].astype(int)

# Calculating Death Density statewise with respect to population
overdose_df['Deaths/Population'] = (overdose_df['Deaths']/overdose_df['Population'])


# In[3]:


# Feature Engineering
# Remove Spacing and Special Characters from Drug Name and Replace it with .
import re
drug_name = opioid_df['Drug Name']
drug_name = drug_name.apply(lambda x:re.sub("\ |-",".",str(x)))
# Considering Features which are related to opioid compounds
opioid_drug_names = set(drug_prescriber_df.columns).intersection(set(drug_name))
opioid_drug_columns = []
# Removing redundant columns
for each in drug_prescriber_df.columns:
    if each in opioid_drug_names:
        pass
    else:
        opioid_drug_columns.append(each)
        
drug_prescriber_df = drug_prescriber_df[opioid_drug_columns]

# Removing Credentials and NPI Column in order to trim our features down.
drug_prescriber_df = drug_prescriber_df.drop(drug_prescriber_df.columns[[0, 3]], axis=1) 

# Convert Categorical Columns
categorical_columns = ['Gender','State','Specialty']

for column in categorical_columns:
    drug_prescriber_df[column] = pd.factorize(drug_prescriber_df[column], sort=True)[0]


# In[4]:


# Train - Test Split
from sklearn.cross_validation import train_test_split
train_df, test_df = train_test_split(drug_prescriber_df, test_size=0.2, random_state=42)
print(train_df.shape)
print(test_df.shape)

#Setting features
features = train_df.iloc[:,0:242]


# ## Hyperparameter Tuning using GridSearchCV and RandomSearchCV  
# <br/>
# <div style="color:blue">
# We will attempt on improving our best 4 classifers using hyperparameter tuning
# </div>

# ### Logistic Regression Parameter Tuning

# In[5]:


## Hyperparameter Tuning using GridSearchCV and RandomSearchCV 
from sklearn.model_selection import GridSearchCV

# Create logistic regression
log_reg_model = LogisticRegression()
# Solver
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# Create regularization hyperparameter space
C = [10**i for i in range(-8,2)]
# Create hyperparameter options
hyperparameters = dict(C=C, solver=solver)
# Create grid search using 5-fold cross validation
clf = GridSearchCV(log_reg_model, hyperparameters, cv=5, verbose=0)
# Fit grid search
best_model = clf.fit(train_df.drop('Opioid.Prescriber',1), train_df['Opioid.Prescriber'])
# View best hyperparameters
print ("Best Parameter Setting:\n")
print('(C = '+str(best_model.best_estimator_.get_params()['C'])+', solver = '+str(best_model.best_estimator_.get_params()['solver'])+")")


# ### KNN Classifier Parameter Tuning

# In[6]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

#Hyper Parameters Set
random_grid = {'n_neighbors':[5,6,7,8,9],
              'leaf_size':[1,2,3,5],
              'weights':['uniform', 'distance'],
              'algorithm':['auto', 'ball_tree','kd_tree','brute'],
              'n_jobs':[-1]}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
knn_model = KNeighborsClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
knn_random = RandomizedSearchCV(estimator = knn_model, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
knn_random.fit(train_df.drop('Opioid.Prescriber',1), train_df['Opioid.Prescriber'])
print(knn_random.best_params_)


# ### Decision Tree Parameter Tuning

# In[7]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_features = randint(1, 9)
min_samples_leaf = randint(1, 9)
min_samples_split = randint(2, 9)
criterion = ["gini", "entropy"]
random_grid = {"max_depth": max_depth,
              "max_features": max_features,
              "min_samples_leaf": min_samples_leaf,
              "min_samples_split":min_samples_split,
              "criterion": criterion}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
decision_tree = DecisionTreeClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
dt_random = RandomizedSearchCV(estimator = decision_tree, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
dt_random.fit(train_df.drop('Opioid.Prescriber',1), train_df['Opioid.Prescriber'])
print(dt_random.best_params_)


# ### Random Forest Parameter Tuning

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
random_forest = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = random_forest, param_distributions = random_grid, n_iter = 5, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_df.drop('Opioid.Prescriber',1), train_df['Opioid.Prescriber'])
print(rf_random.best_params_)


# In[8]:


from sklearn.model_selection import KFold

# Takes as inputs a dataset, a label name, # of splits/folds (k), a sequence of values for  CC  (cs)
def xValSVM(dataset, label_name, k, cs, model):

    # Removing Target Variable from dataset and storing it seperately
    data = dataset.drop(label_name, axis=1)
    target = dataset[label_name]
    aucs = {}
    
    for f in range(2,k):
        cross_val = KFold(n_splits = f)
        
        # Split the data into data_train & data_validate 
        for data_train_index, data_validate_index in cross_val.split(X=data, y=target):
            data_train = dataset.iloc[data_train_index]
            data_validate = dataset.iloc[data_validate_index]
            
            for c in cs:
                # Fit the Model
                model.fit(data_train.drop(label_name, axis=1),data_train[label_name])
                # Predicting Labels, Calculating FPR, TPR, Thresholds and ROC Value using SVM Model
                predictions = model.predict_proba(data_validate.drop(label_name, axis=1))[:, 1]
                fpr, tpr, thresholds = metrics.roc_curve(data_validate[label_name], predictions)
                # Computes AUC_c_k on validation data
                AUC_c_k = metrics.roc_auc_score(data_validate[label_name], predictions)
                # Stores AUC_c_k in a dictionary of values
                if c in aucs:
                    aucs[c].append(AUC_c_k)
                else:
                    aucs[c] = [AUC_c_k]
    
    # Returns a dictionary, where each key-value pair is: c:[auc-c1,auc-c2,..auc-ck]
    return aucs


# In[9]:


# Generate a sequence of 10C  values in the interval [10^(-8), ..., 10^1] (i.e., do all powers of 10 from -8 to 1).
cs = [10**i for i in range(-8,2)]

log_reg_model=LogisticRegression(random_state=22,C=0.0001,solver='newton-cg',max_iter=200)
knn_model = KNeighborsClassifier()
decision_tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 110, max_features = 8,                                       min_samples_leaf = 3, min_samples_split = 2)
random_forest = RandomForestClassifier(bootstrap = False, max_depth = 80, max_features = 'auto',                                        min_samples_leaf = 1, min_samples_split = 10, n_estimators = 1000)

Name=[]
Model_AUC=[]

for model, label in zip([log_reg_model, knn_model, decision_tree, random_forest],                         ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest']):
    # Cross-Validation
    aucs = xValSVM(train_df, 'Opioid.Prescriber', 10 , cs, model)
    mean_auc = pd.DataFrame(aucs).mean().mean()
    max_auc = pd.DataFrame(aucs).max().max()
    Model_AUC.append(mean_auc)
    Name.append(model.__class__.__name__)
    print("Mean AUC achieved: %f of model %s" % (mean_auc,label))
    print("Max AUC achieved: %f of model %s" % (max_auc,label))

