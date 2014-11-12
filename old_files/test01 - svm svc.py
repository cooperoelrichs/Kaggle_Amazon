"""
================================
test: SVM on amazon data
================================

based on: `using_kernels_tut`
http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html#support-vector-machines-svms
"""
print __doc__

import numpy as np
import pylab as pl
import time as time
from sklearn import datasets, svm

dir = 'D:\Cooper\Google Drive\Kaggle_Amazon'
file = 'train_reduced.csv'

file_start = time.clock()
with open(dir + '\\' + file) as f:
    lines = f.read().splitlines()

n_sample = len(lines)
n_dim = len(lines[0].split(','))

target = np.zeros(n_sample)
data = np.zeros((n_sample, n_dim-1))

for n,l in enumerate(lines):
    if n == 0: 
        print(l)
        continue
    l = l.split(',')
    for m,i in enumerate(l):
        if m == 0: target[n] = i
        if m > 0: data[n, m-1] = i

file_time = time.clock() - file_start
print(file + ' read and process time (s): ' + str(file_time))

np.random.seed(0)
order = np.random.permutation(n_sample)
X = data[order]
y = target[order].astype(np.float)

X_train = X[: .9 * n_sample]
y_train = y[: .9 * n_sample]
X_test = X[.9 * n_sample :]
y_test = y[.9 * n_sample :]

# fit the model
for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    learn_start = time.clock()
    
    print('processing: ' + kernel)
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)
    
    learn_time = time.clock() - learn_start
    print(kernel + ' learn time (s): ' + learn_time)
    print(kernel + ' score: %f' % clf.fit(X_train, y_train).score(X_test, y_test))
pl.show()