from sklearn import (metrics, cross_validation, preprocessing)
from csv_reader import csvReader
from skl_preped_models import (skl_models, param_grid_dict)
from multiprocessing import Pool
import numpy as np
import itertools as it
import time

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

def setup_and_fit_model(data): #model, X, y, X_cv, y_cv, score_func, test_params):
    model, X, y, X_cv, y_cv, score_func, test_params = data
    model.set_params(**test_params)
    model.fit(X, y) 
    score = score_func(y_cv, model.predict_proba(X_cv)[:, 1])
    return([score, test_params])

def get_best_results(results):
    best_score = 0
    for score, params in results:
        if score > best_score: best_score, best_params = score, params
    return(best_score, best_params)

def process_results(results, keys):
    processed_params = ''
    best_score, best_params = get_best_results(results)
    for i, key in enumerate(keys):
        processed_params += key + ': ' + str(best_params[key])
        if i+1 != len(keys): processed_params += ', '
    return([best_score, processed_params])
    

def hyper_parameterisator(model, X, y, X_cv, y_cv, params_dicts, score_func=metrics.auc_score, n_jobs=8 ):
    print('Starting grid search')
    scores, keys, param_list, setup_and_fit = [], [], [], []
    for params_dict in params_dicts:
        for key, params in params_dict.items():
            keys.append(key)
            param_list.append(params)
        param_grid = list(it.product(*param_list))
        
        for params in param_grid:
            test_params = {}
            for i, param in enumerate(params):
                test_params[keys[i]] = param
            setup_and_fit.append([model, X, y, X_cv, y_cv, score_func, test_params])
    
    pool    = Pool(processes=n_jobs)
    results = pool.map(setup_and_fit_model, setup_and_fit)
    
    return(process_results(results, keys))

if __name__ == '__main__':
    directory   = 'D:\Cooper\Google Drive\Kaggle_Amazon\data' #WINDOWS
    #directory   = 'C:\Google Drive\Kaggle_Amazon\data' #WINDOWS 2
    #directory   = '/Users/cooperoelrichs/Google Drive/Kaggle_Amazon/data' #MAC
    training    = 'train.csv'
    testing     = 'test.csv'
    results     = 'results.csv'
    
    y_train, X_train, index_test, X_test  = get_data(directory, training, testing)
    
    models, data, seed = [], [], 105
    for i, (model_name, model) in enumerate(skl_models):
        if model_name != 'svc_rbf': continue
        #if i > 0: continue
        cv = cross_validation.train_test_split(X_train, y_train, test_size=.20, random_state=seed)
        X_train_cv, X_test_cv, y_train_cv, y_test_cv = cv
        
        t_now = time.time()
        res = hyper_parameterisator(model, X_train_cv, y_train_cv, X_test_cv, y_test_cv, param_grid_dict[model_name])
        roc_auc, best_params = res ; elapsed = time.time() - t_now
        print "AUC (model: %s, t: %.2f min): %f" % (model_name, elapsed/60, roc_auc)
        print("-- Best parameters - %s" % best_params)

