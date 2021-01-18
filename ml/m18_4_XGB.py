from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=66, shuffle=True, test_size=0.2)

print(x_train.shape)

# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier(max_depth=4)
# model = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(max_depth=4)


model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)
print(model.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model) :
    n_features=cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()







'''
0.9736842105263158
[0.         0.03518598 0.00053468 0.02371635 0.00661651 0.02328466
 0.00405836 0.09933352 0.00236719 0.         0.01060954 0.00473884
 0.01074011 0.01426315 0.0022232  0.00573987 0.00049415 0.00060479
 0.00522006 0.00680739 0.01785728 0.0190929  0.3432317  0.24493258
 0.00278067 0.         0.01099805 0.09473949 0.00262496 0.00720399]
 '''

