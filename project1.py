# Random Forest Classification

import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             roc_auc_score,
                             roc_curve,
                             confusion_matrix)
from sklearn.model_selection import (cross_val_score,
                                     GridSearchCV,
                                     RandomizedSearchCV,
                                     learning_curve,
                                     validation_curve,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from warnings import filterwarnings


# Importing the dataset
df = pd.read_csv('C:/Users/dell/Desktop/hr_data.csv')

# Map salary into integers
salary_map = {"low": 0, "medium": 1, "high": 2}
df["salary"] = df["salary"].map(salary_map)

# Map salary into integers
department_map = {"sales": 0, "accounting": 1, "hr": 2, "technical": 3, "support": 4,"management": 5, "IT": 6, "product_mng": 7, "marketing": 8, "RandD": 9}
df["department"] = df["department"].map(department_map)
df.head()

# Convert dataframe into numpy objects and split them into
# train and test sets: 80/20
X = df.loc[:, df.columns != "left"].values
y = df.loc[:, df.columns == "left"].values.flatten()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1)

# Upsample minority class
X_train_u, y_train_u = resample(X_train[y_train == 1],
                                y_train[y_train == 1],
                                replace=True,
                                n_samples=X_train[y_train == 0].shape[0],
                                random_state=1)
X_train_u = np.concatenate((X_train[y_train == 0], X_train_u))
y_train_u = np.concatenate((y_train[y_train == 0], y_train_u))

# Downsample majority class
X_train_d, y_train_d = resample(X_train[y_train == 0],
                                y_train[y_train == 0],
                                replace=True,
                                n_samples=X_train[y_train == 1].shape[0],
                                random_state=1)
X_train_d = np.concatenate((X_train[y_train == 1], X_train_d))
y_train_d = np.concatenate((y_train[y_train == 1], y_train_d))

# Reassign original training data to upsampled data
X_train, y_train = np.copy(X_train_u), np.copy(y_train_u)

# Delete original and downsampled data
del X_train_u, y_train_u, X_train_d, y_train_d

# Refit RF classifier using best params
clf_rf = make_pipeline(StandardScaler(),
                       RandomForestClassifier(n_estimators=50,
                                              criterion="entropy",
                                              max_features=0.4,
                                              min_samples_leaf=1,
                                              class_weight="balanced",
                                              n_jobs=-1,
                                              random_state=123))


clf_rf.fit(X_train, y_train)
import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(clf_rf, file)

# !/usr/bin/python3
from tkinter import *

root = Tk()

#changing the title of our master widget      
root.title("Employee Turnover Predictor")

v1 = StringVar()
v2 = StringVar()
v3 = StringVar()
v4 = StringVar()
v5 = StringVar()
v6 = StringVar()
v7 = StringVar()
v8 = StringVar()
v9 = StringVar()

    
label1 = Label( root, text = "Satisfaction Level(0-1): ").grid(row=0, column=0,sticky=W,ipady=5)
entry1 = Entry(root, textvariable = v1).grid(row=0, column=1)

label2 = Label( root, text = "Last Evaluation: ").grid(row=1, column=0,sticky=W,ipady=5)
entry2 = Entry(root, textvariable = v2).grid(row=1, column=1)

label3 = Label( root, text = "Project Number: ").grid(row=2, column=0,sticky=W,ipady=5)
entry3 = Entry(root, textvariable = v3).grid(row=2, column=1)

label4 = Label( root, text = "Average Monthly Hours: ").grid(row=3, column=0,sticky=W,ipady=5)
entry4 = Entry(root, textvariable = v4).grid(row=3, column=1)

label5 = Label( root, text = "Time Spent in company(in years): ").grid(row=4, column=0,sticky=W,ipady=5)
entry5 = Entry(root, textvariable = v5).grid(row=4, column=1)

label6 = Label( root, text = "Work Accident(0 or 1): ").grid(row=5, column=0,sticky=W,ipady=5)
entry6 = Entry(root, textvariable = v6).grid(row=5, column=1)

label7 = Label( root, text = "Promotion in last 5 years(0 or 1):").grid(row=6, column=0,sticky=W,ipady=5)
entry7 = Entry(root, textvariable = v7).grid(row=6, column=1)

label8 = Label( root, text = "Name of department: ").grid(row=7, column=0,sticky=W,ipady=5)
entry8 = Entry(root, textvariable = v8).grid(row=7, column=1)

label9 = Label( root, text = "Salary(low/medium/high): ").grid(row=8, column=0,sticky=W,ipady=5)
entry9 = Entry(root, textvariable = v9).grid(row=8, column=1)

label13 = Label(root,text=" ").grid(row=9,column=0)


def fun():
    list1 = []
    satisfaction_level = float(v1.get())
    last_evaluation = float(v2.get())
    number_project = float(v3.get())
    average_montly_hours = float(v4.get())
    time_spend_company = float(v5.get())
    Work_accident = float(v6.get())
    promotion_last_5years = float(v7.get())
    department = float(v8.get())
    salary = float(v9.get())

    list1.append(satisfaction_level)
    list1.append(last_evaluation)
    list1.append(number_project)
    list1.append(average_montly_hours)
    list1.append(time_spend_company)
    list1.append(Work_accident)
    list1.append(promotion_last_5years)
    list1.append(department)
    list1.append(salary)
    
    ind = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'department', 'salary']
    test_x1 = pd.DataFrame(list1, index = ind)
    
    
    y_pred = clf_rf.predict(test_x1.T)

    if(y_pred == 0):
        result = 'Employee will not leave!'
    else:
        result = 'Employee will leave!'
    
    label10 = Label(root, text="Prediction: "+result ).grid(row=11,column=0,sticky=W,ipady=5, columnspan=2)
    
button1 = Button(root, text="Predict it!",width=10, bg='#2ff', command = fun).grid(row=10,column=0,columnspan=2,ipady=5)
label11 = Label(root, text="  ").grid(row=11,column=0,sticky=W,ipady=5, columnspan=2)

                 
root.mainloop()