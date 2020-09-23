import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

features1 = pd.read_csv('../Dataset/Sample1.csv')
features1.head()
features2 = pd.read_csv('../Dataset/Sample2.csv')
features2.head()
features = pd.concat([features1, features2])
features.head()
# print(features)

features = features.replace('mod', 0)
features = features.replace('unm', 1)
features = features.replace(np.nan, 0, regex=True)

# print(features)
X = features[['q1', 'q2', 'q3', 'q4', 'q5', 'mis1', 'mis2', 'mis3', 'mis4', 'mis5']].astype(float)
Y = features['sample'].astype(int)

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=True)

from RNA1.minisom import MiniSom

som = MiniSom(x=10, y=10, input_len=10, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

from pylab import bone, pcolor, colorbar, plot, show

bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
Y = Y[:, ]
print(Y)
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[Y[i]],
         markeredgecolor=colors[Y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

print('done')
# clf = RandomForestClassifier(n_estimators=80)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
# sn.heatmap(confusion_matrix, annot=True)

# print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
# plt.show()
