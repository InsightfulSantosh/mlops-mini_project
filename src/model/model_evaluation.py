import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score ,roc_auc_score
import logging
import mlflow
import pickle
import json
import dagshub

file_path= './models/model.pkl'
with open (file_path,'rb') as file:
    model=pickle.load(file)

data_path='./data/processed/test_bow.csv'
data=pd.read_csv(data_path)
xtest= data.iloc[:,:-1].values
prediction=model.predict_proba(xtest)
print(prediction)