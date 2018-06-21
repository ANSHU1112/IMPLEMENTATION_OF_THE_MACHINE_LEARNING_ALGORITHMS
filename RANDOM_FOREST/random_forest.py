import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('/home/priyanshu/Desktop/PRIYANSHU/MAIN/LAB/breast.csv')
data.head()

colnames=['ID', 'RADIUS', 'TEXTURE', 'PERIMETER', 'AREA', 'SMOOTHNESS', 'COMPACTNESS', 'CONCAVITY', 'CONCAVE', 'SYMMETRY', 'FRACTAL']
data = pd.read_csv('/home/priyanshu/Desktop/PRIYANSHU/MAIN/LAB/breast.csv', names=colnames, header=None)
data.head()

from sklearn.cross_validation import train_test_split
X = data.iloc[0:, [1,2,3,4,5,6,7,8,9]].values
X_train,X_test,Y_train,Y_test = train_test_split(X,data['FRACTAL'], test_size=0.3, random_state=0)

rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train, Y_train)
accuracy = rf.score(X_test, Y_test)
print("Accuracy = {}% ".format(accuracy*100))

X = [[6,2,4,1,4,6,4,5,2]]
rf.predict(X)

