from sklearn import (metrics, cross_validation, preprocessing)
from sklearn.grid_search import GridSearchCV
from csv_reader import csvReader
from skl_preped_models import (skl_models, param_grid_dict)
from multiprocessing import Pool
import numpy as np

def single_cv(data):
    model, X_train, y_train, seed = data
    cv = cross_validation.train_test_split(X_train, y_train, test_size=.20, random_state=seed)
    X_train_cv, X_test_cv, y_train_cv, y_test_cv = cv
    
    model.fit(X_train_cv, y_train_cv) 
    preds = model.predict_proba(X_test_cv)[:, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_test_cv, preds)
    roc_auc = metrics.auc(fpr, tpr)
    return(roc_auc)

def one_hot_encoding(X_train, X_test):
    print('One Hot Encoding')
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X_train, X_test)))
    X_train = encoder.transform(X_train)  # Returns a sparse matrix (see numpy.sparse)
    X_test  = encoder.transform(X_test)
    return(X_train, X_test)

def get_data(directory, training, testing):
    train_file      = directory + '/' + training
    test_file       = directory + '/' + testing
    
    y_train, X_train    = csvReader().get_train(fname = train_file)
    index_test, X_test  = csvReader().get_test(fname = test_file)
    X_train, X_test     = one_hot_encoding(X_train, X_test)
    return([y_train, X_train, index_test, X_test])

if __name__ == '__main__':
    directory   = 'D:\Cooper\Google Drive\Kaggle_Amazon\data' #WINDOWS
    #directory   = '/Users/cooperoelrichs/Google Drive/Kaggle_Amazon/data' #MAC
    training    = 'train.csv'
    testing     = 'test.csv'
    results     = 'results.csv'
    
    y_train, X_train, index_test, X_test  = get_data(directory, training, testing)
    
    models, data, seed = [], [], 105
    for i, (model_name, model) in enumerate(skl_models):
        if i > 1: continue
        cv = cross_validation.train_test_split(X_train, y_train, test_size=.20, random_state=seed)
        X_train_cv, X_test_cv, y_train_cv, y_test_cv = cv
        
        print('Starting grid search')
        clf = GridSearchCV(model, param_grid_dict[model_name], score_func=metrics.auc_score, n_jobs=10)
        clf.fit(X_train_cv, y_train_cv, cv=3)
        preds = clf.predict_proba(X_test_cv)[:, 1]
        roc_auc = metrics.auc_score(y_test_cv, preds)
        print("Best parameters: %s" % clf.best_params_)
        print "AUC (model: %s): %f" % (model_name, roc_auc)
        
        #for j in range(4): data.append([model, X_train, y_train, j*seed])
        #pool = Pool()
        #rocs = pool.map(single_cv, data)
        #print "AUC (model %d/%d): %f" % (i + 1, len(skl_models), np.mean(rocs))



