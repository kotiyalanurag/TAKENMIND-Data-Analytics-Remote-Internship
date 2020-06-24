import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

df_iris = pd.read_csv('IrisD.csv')
df_iris.head()

df_iris.drop('Id', 1, inplace=True)

df_X = df_iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df_iris['Species']

X = np.asarray(df_X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf.fit(X_train, y_train)

pred_tree = clf.predict(X_test)

print("Decision Tree's Accuracy : ", metrics.accuracy_score(y_test, pred_tree))
