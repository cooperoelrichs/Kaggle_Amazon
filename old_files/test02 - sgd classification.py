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
from sklearn.linear_model import SGDClassifier, preprocessing

from csv_processor import csvProcessor

dir = 'D:\Cooper\Google Drive\Kaggle_Amazon'
training = 'train.csv'
testing = 'test.csv'

training_csv = csvProcessor(dir, training)
training_csv.read_csv(True)
data_sets = training_csv.randomise_csv()
X_train, y_train, X_test, y_test = data_sets

testing_csv = csvProcessor(dir, testing)
testing_data = testing_csv.read_csv(False)

# Fit the model
learn_start = time.clock()

print('processing')
clf = SGDClassifier(alpha=0.1, n_iter=20) #, n_jobs=-1)
clf.fit(X_train, y_train)

learn_time = time.clock() - learn_start
print('learn time (s): ' + str(learn_time))

# Show results
print('coef_: ' + str(clf.coef_))
print('intercept: ' + str(clf.intercept_))
print('Score: %f' % clf.score(X_test, y_test))

results = np.zeros((len(testing_data),2))

test = [
        [1,0,36,117961,118413,119968,118321,117906,290919,118322],
        [1,15677,16569,117961,118327,120559,122269,141096,118643,122271],
        [1,19303,19642,117961,118413,118481,118784,240983,290919,118786],
        [1,34924,16850,117961,118225,119238,122849,149223,119095,122850],
        [1,41569,8204,117961,118052,122392,120097,174445,270488,120099],
        [1,75241,15781,117961,118300,118395,118890,305057,118398,118892],
        [1,81520,1350,117961,118052,122938,117905,117906,290919,117908],
        [1,189629,16922,117961,117962,118501,118784,124402,290919,118786],
        [1,73756,77555,126918,126919,119136,119065,169395,118667,119067],
        [1,933,7578,117961,118343,120722,118321,117906,290919,118322],
        [0,45333,14561,117951,117952,118008,118568,118568,19721,118570,],
        [0,7678,51172,117961,118225,120551,120690,120691,290919,120692,],
        [0,6042,8088,117961,117962,118481,117905,117906,290919,117908,],
        [0,33054,16830,117961,118327,119830,120773,118959,118960,120774,],
        [0,20231,5559,119280,119281,118623,118995,286106,292795,118997,],
        ]

for i,y in enumerate(test):
    x = clf.predict(y[1:])
    print(str(y[0]) + ', ' + str(x))

#for i,y in enumerate(testing_data):
#    x = clf.predict(y[1:])
#    results[i, 0] = y[0]
#    results[i, 1] = x
    
#training_csv.write_csv(dir + '\\' + 'results.csv', results, ['ID','ACTION'])