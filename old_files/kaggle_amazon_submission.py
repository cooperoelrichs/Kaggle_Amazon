from sklearn import (preprocessing)
from csv_reader import csvReader
from skl_preped_models import (submission_models)
import numpy as np
import time

def one_hot_encoding(X_train, X_test=None):
    encoder = preprocessing.OneHotEncoder()
    
    if X_test == None:
        encoder.fit(X_train)
        X_train = encoder.transform(X_train)
    else:
        encoder.fit(np.vstack((X_train, X_test)))
        X_train = encoder.transform(X_train)
        X_test  = encoder.transform(X_test)
        return(X_train, X_test)
        return(X_train)

def get_data(directory, training, testing):
    train_file      = directory + '/' + training
    test_file       = directory + '/' + testing
    
    y_train, X_train    = csvReader().get_train(fname = train_file)
    index_test, X_test  = csvReader().get_test(fname = test_file)
    
    features = [0,1,3,4,6,8]
    X_train = X_train[:, features]
    X_test  = X_test[:, features]
    
    X_train, X_test     = one_hot_encoding(X_train, X_test)
    return([y_train, X_train, index_test, X_test])

def save_results(predictions, filename, directory):
    """Given a vector of predictions, save results in CSV format."""
    with open(directory + '/' + filename, 'w') as f:
        f.write("id,Action\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))
    print('Results saved to: %s, in: %s' %(filename, directory))


if __name__ == '__main__':
    directory   = 'D:\Cooper\Google Drive\Kaggle_Amazon\data' #WINDOWS
    #directory   = 'C:\Google Drive\Kaggle_Amazon\data' #WINDOWS 2
    #directory   = '/Users/cooperoelrichs/Google Drive/Kaggle_Amazon/data' #MAC
    training    = 'train.csv'
    testing     = 'test.csv'
    results     = 'results.csv'
    
    y_train, X_train, index_test, X_test  = get_data(directory, training, testing)
    
    models, data, seed = [], [], 18
    for i, (model_name, model) in enumerate(submission_models):
        #if model_name != 'svc_rbf': continue
        if i > 0: continue
        t_now = time.time()
        print('Fitting model: %s' % model_name)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        save_results(preds, 'results_%s_%i.csv' %(model_name, i), directory)
        

