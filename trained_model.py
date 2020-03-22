"""
Features are the symptoms for the infection. People can decide better and valuable features.
Features : collected from the WHO website and generated random data for those features.

Fever - continous variable
Tiredness - binary variable
Dry_Cough - binary variable
Aches_Pains - binary variable
Nasal_Congestion - binary variable
Runny_Nose - binary variable
Sore_throat - binary variable
Diahrea - binary variable
Shortness_Of_Breath - binary variable
Age - Discrete Variable
Ouput : probability of having infection based on the features

### install
!pip3 install pandas sklearn numpy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

data = pd.read_csv("Data - Sheet1.csv")
train, test = train_test_split(data, test_size=0.2)
test = test.to_numpy()
train = train.to_numpy()
X_train = train[:, :-1]
X_test = test[:, :-1]
Y_train = train[:,-1]
Y_test = test[:,-1]


clf = SVC(gamma='auto', probability=True)
clf.fit(X_train, Y_train)

model_file =  open("model.pkl", "wb")
pickle.dump(clf, model_file)
model_file.close()
