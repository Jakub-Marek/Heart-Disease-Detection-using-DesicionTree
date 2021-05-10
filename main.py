"""
Authors: Riken Patel & Jakub Marek
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pydot

## Loading in the data from excel spreadsheet for heart disease UCI:
data = pd.read_excel('heart.xlsx')
attributes = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
labels = np.array(data['target'])
x = data[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]

#Sort Data into test and trian samples
x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size = 0.2, random_state=0)


## Obtaining parameters (GridSearchCV) DecisionTreeClassifier
clf_df = DecisionTreeClassifier()
df_parameters = {'criterion': ['entropy','gini'],'max_depth': [5,10,None],
                             'max_leaf_nodes': [2,5],'min_samples_split': [2,5,10]}
scorer_df  = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
clf_GS_df = GridSearchCV(clf_df, df_parameters,scoring=scorer_df, refit='Accuracy', return_train_score=True)
grid_obj = clf_GS_df.fit(x_train, labels_train)
best_params_df = grid_obj.best_params_
print(f"Best params for DecisionTreeClassifier: {best_params_df}")

## Applying the parameters into DecisionTreeClassifier
clf_best_df = grid_obj.best_estimator_
df_fit = clf_best_df.fit(np.array(x_train),labels_train)
pred_df = clf_best_df.predict(np.array(x_test))
accuracy = accuracy_score(labels_test, pred_df)
print("Accuracy of Decision Tree Classifier:",accuracy*100, '%')
tree.plot_tree(df_fit)
plt.show()


## Obtaining parameters (GridSearchCV) RandomForestClassifier
clf_rf = RandomForestClassifier()
rf_parameters = {'n_estimators': [10,15,20],
                'max_depth': [10,None],'random_state': [12,5,None]}
scorer_rf  = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
clf_GS_rf = GridSearchCV(clf_rf, rf_parameters,scoring=scorer_rf,refit='Accuracy', return_train_score=True)
grid_obj = clf_GS_rf.fit(x_train, labels_train)
best_params = grid_obj.best_params_
print(f"Best params for RandomForestClassifier: {best_params}")

## Applying the parameters into RandomForestClassifier
clf_best_rf = grid_obj.best_estimator_
rf_fit = clf_best_rf.fit(np.array(x_train),labels_train)
pred_rf = clf_best_rf.predict(np.array(x_test))
accuracy = accuracy_score(labels_test, pred_rf)
print("Accuracy of Random Forest Classifier:",accuracy*100, '%')

##Plotting the Random Forest using Graphviz
tree_layer = clf_best_rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree_layer, out_file = 'tree.dot', feature_names = attributes, rounded = True)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')
