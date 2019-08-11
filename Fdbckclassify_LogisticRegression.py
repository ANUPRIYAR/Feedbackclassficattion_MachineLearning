# coding: utf-8
"""
Author: Anupriya Ramachandran
Classifies the feedback on Remedy Interface from different Organisation Units
into different categories based on prediction probabilities through Logistic
Regression
Output: Written to text file 'summary.txt'
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# Data in csv format with utf-8 encoding
train = pd.read_csv('C:/Users/AR049995/Desktop/Input_files/Train/Traindata.csv')
test = pd.read_csv('C:/Users/AR049995/Desktop/Input_files/Test/Testdata.csv')

label_columns = ['Complicated_UI', 'Poor_performance', 'Service_unavailability']

# To replace Null values in dataframe
COMMENT = 'Additional_Comments'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

# Building a model
n = train.shape[0]
vec = CountVectorizer(ngram_range=(1, 2), stop_words="english")
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])


# Naive Bayes feature equation
def pr(y_i, y):
    p = x[y == y_i].sum(0)
    return (p + 1) / ((y == y_i).sum() + 1)


x = trn_term_doc.sign()
test_x = test_term_doc.sign()


def get_mdl(y):
    y = y.values
    r = np.log(pr(1, y) / pr(0, y))
    m = LogisticRegression(C=0.1, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


# Initiate a matrix
preds = np.zeros((len(test), len(label_columns)))

# Create a matrix with probabilities
for i, j in enumerate(label_columns):
    print('fit', j)
    m, r = get_mdl(train[j])
    preds[:, i] = m.predict_proba(test_x.multiply(r))[:, 1]

submid = pd.DataFrame({'Additional_Comments': test["Additional_Comments"]})
length = len(preds)
output = np.ones((len(test), len(label_columns)))

for i in range(0, length):
    for j in range(0, 3):
        if preds[i][j] <= 0.4:
            preds[i][j] = 0
        else:
            preds[i][j] = 1

submission = pd.concat([submid, pd.DataFrame(preds,
                                             columns=
                                             label_columns)], axis=1)
submission.to_csv('C:/Users/AR049995/Desktop/Input_files/data.csv', index=False)

data = pd.read_csv('C:/Users/AR049995/Desktop/Input_files/data.csv')
org_data = pd.read_csv('C:/Users/AR049995/Desktop/Input_files/AITest.csv')

data['Org_Unit'] = org_data['Org_Unit']
labels = ['Complicated_UI', 'Poor_performance', 'Service_unavailability']


def calculate_percent(data, label):
    sum_label = data[label].sum()
    total = data[label].size
    percent = (sum_label * 100) / total
    return percent


list_Org = data['Org_Unit'].unique().tolist()

with open('summary.txt', 'w') as file:
    file.write("CUMULATIVE FEEDBACK ON REMEDY INTERFACE FROM "
               "DIFFERENT "
               "ORGANISATION UNITS IN CERNER \n")
    file.write(
        "===================================================================\n")
    for org in list_Org:
        Org_data = data[data['Org_Unit'] == org]
        for label in labels:
            total = calculate_percent(Org_data, label=label)
            if total != 0.0:
                print("%s percent people in %s ,feel that remedy has %s"
                      % (total, org, label))
                file.write("%s percent people in '%s' Organisation ,feel that "
                           "remedy has "
                           "'%s' \n"
                           % (total, org, label))
