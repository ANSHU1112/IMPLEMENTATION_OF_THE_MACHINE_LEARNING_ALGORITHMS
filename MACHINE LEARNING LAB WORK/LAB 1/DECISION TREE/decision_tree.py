
# coding: utf-8

# In[1]:

#DATA PREPROCESSING


# In[2]:

import pandas as pd


# In[3]:

data = pd.read_csv('C:/Users/PRIYANSHU SHARMA/Desktop/PRIYANSHU/6 STUDY/6 SEMSTER/MACHINE LEARNING/LAB/breast.csv')


# In[4]:

data.head()


# In[5]:

colnames=['ID', 'RADIUS', 'TEXTURE', 'PERIMETER', 'AREA', 'SMOOTHNESS', 'COMPACTNESS', 'CONCAVITY', 'CONCAVE', 'SYMMETRY', 'FRACTAL']
data = pd.read_csv('C:/Users/PRIYANSHU SHARMA/Desktop/PRIYANSHU/6 STUDY/6 SEMSTER/MACHINE LEARNING/LAB/breast.csv', names=colnames, header=None)


# In[6]:

data.head()


# In[7]:

print(data.columns)


# In[8]:

data.describe()


# In[9]:

get_ipython().magic('matplotlib inline')


# In[10]:

import matplotlib as plt
import seaborn as sb


# In[11]:

cols = ['RADIUS', 'TEXTURE', 'PERIMETER', 'AREA'] #just only setting the variable


# In[12]:

sb.pairplot(data.dropna(),hue='FRACTAL')


# In[13]:

data['FRACTAL'].unique() #WE CAN MAKE THE USE OF UNIQUE FUNCTION IN ORDER TO SEE WHICH ATTRIBUTE IS BETTER FOR CLASSIFICATION


# In[14]:

data['RADIUS'].unique()


# In[15]:

data['PERIMETER'].unique()


# In[16]:

data.loc[(data['CONCAVE']),'FRACTAL'].hist()


# In[17]:

data['CONCAVE'].unique()


# In[18]:

data.loc[(data['RADIUS']),'FRACTAL'].hist()


# In[19]:

data.loc[(data['FRACTAL']),'RADIUS'].hist()


# In[20]:

data.loc[(data['RADIUS']==1) ]


# In[21]:

data['SYMMETRY'].unique()


# In[22]:

data.loc[(data['SYMMETRY']==5) ]


# In[23]:

data.loc[(data['RADIUS']==1),'FRACTAL'].mean()


# In[24]:

import matplotlib.pyplot as plt


# In[25]:

plt.figure(figsize=(10, 10))
cols =['RADIUS', 'TEXTURE', 'PERIMETER', 'AREA', 'SMOOTHNESS', 'COMPACTNESS', 'CONCAVITY', 'CONCAVE', 'SYMMETRY', 'FRACTAL'] 
for column_index, column in enumerate(data[cols].columns):
    if column == 'FRACTAL':
        continue
    plt.subplot(3, 3, column_index + 1)
    sb.violinplot(x='FRACTAL', y=column, data=data[cols])


# In[37]:

plt.figure(figsize=(10, 10))
for column_index, column in enumerate(data[cols].columns):
    if column == 'FRACTAL':
        continue
    plt.subplot(3, 3, column_index + 1)
    sb.boxplot(x='FRACTAL', y=column, data=data[cols])


# In[26]:

# DECISION TREE IMPLEMENTATION


# In[27]:

from sklearn.tree import DecisionTreeClassifier


# In[28]:

from sklearn.model_selection import train_test_split


# In[29]:

print(data.columns)


# In[30]:

columns=['RADIUS', 'TEXTURE', 'PERIMETER', 'AREA', 'SMOOTHNESS', 'COMPACTNESS', 'CONCAVITY', 'CONCAVE', 'SYMMETRY']
a=data[columns].iloc[:,:9].values #all columns in array
a


# In[31]:

b=data[columns].iloc[:,0:1].values #label column in array (particular column selection)
b


# In[32]:

X_train,X_test,Y_train,Y_test = train_test_split(data[columns],data['FRACTAL'],test_size=0.4,random_state=14)
tree = DecisionTreeClassifier(max_depth=7,random_state=0)
tree.fit(X_train,Y_train)


# In[33]:

print("Accuracy on the training set: %.3f" % tree.score(X_train,Y_train))
print("Accuracy on the testing set: %.3f" % tree.score(X_test,Y_test))


# In[34]:

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=['2','4'], impurity=False, filled=True,
                feature_names=data[columns].columns)


# In[35]:

import graphviz


# In[36]:

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

