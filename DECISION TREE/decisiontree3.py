import pandas as pd

from sklearn import tree
iris = read_csv('C:\Users\PRIYANSHU SHARMA\Downloads\bank.csv')
clf = tree.DecisionTreeClassifier(criterion = "entropy")
clf = clf.fit(iris.data, iris.target)

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names,  class_names=iris.target_names,  filled=True, rounded=True,  special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.view()
clf.predict(iris.data[:1, :])
clf.predict_proba(iris.data[:1, :])