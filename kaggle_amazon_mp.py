from sklearn import (metrics, cross_validation, )
from skl_preped_models import (skl_models, paramaters, selected_features, greedy_features, param_grid_dict)
from multiprocessing import Pool
import numpy as np
import itertools as it
import time
import cPickle as pickle
import copy

from ml_helpers import *

def single_cv(data):
    model, X_train, y_train, seed = data
    cv = cross_validation.train_test_split(X_train, y_train, test_size=.20, random_state=seed)
    X_train_cv, X_test_cv, y_train_cv, y_test_cv = cv
    
    model.fit(X_train_cv, y_train_cv) 
    preds = model.predict_proba(X_test_cv)[:, 1]

    fpr, tpr, _ = metrics.roc_curve(y_test_cv, preds)
    roc_auc = metrics.auc(fpr, tpr)
    return(roc_auc)

def setup_and_fit_model(data): #model, X, y, X_cv, y_cv, score_func, test_params):
    model, X, y, score_func, test_params, f_set, cv_iter = data
    #model.set_params(**test_params)
    sum_score = 0
    for i in range(cv_iter):
        cv = cross_validation.train_test_split(X, y, test_size=.20, random_state=46*(i+1))
        sum_score += model_test([model, cv, score_func, test_params])
    return([sum_score/cv_iter, test_params, f_set])

def model_test(data):
    model, cv, score_func, test_params = data
    X_cv, X_test_cv, y_cv, y_test_cv = cv
    model.set_params(**test_params)
    model.fit(X_cv, y_cv)
    score = score_func(y_test_cv, model.predict_proba(X_test_cv)[:, 1])
    return(score)

def multiprocessor(func, args, processors):
    tasks, time_start = len(args), time.time()
    if tasks > 12: print('Starting pooled operation of %i tasks' % tasks)
    pool    = Pool(processes=processors)
    results = pool.map_async(func, args)
    pool.close()
    percent_last = 0
    while (True):
        if (results.ready()): break
        remaining = results._number_left
        time_elapsed = (time.time() - time_start)/60
        percent = (float(tasks - remaining) / float(tasks) * 100)
        if percent - percent_last > 20:
            if tasks > 12: print("      %.1f min - %i remaining - %.0f%%" % (time_elapsed, remaining, percent))
            percent_last = percent
        time.sleep(1)
    return(results.get())

def hyper_parameterisator(model, X, y, params_dicts, cv_iter, score_func=metrics.auc_score, n_jobs=None, f_set = ['all']):
    print('Starting grid search')
    keys, param_list, setup_and_fit = [], [], []
    if f_set[0] == 'all': X = one_hot_encoding(X)
    else:                 X = one_hot_encoding(X[:,f_set])
    
    for params_dict in params_dicts:
        for key, params in params_dict.items():
            keys.append(key)
            param_list.append(params)
        param_grid = list(it.product(*param_list))
        
        for params in param_grid:
            test_params = {}
            for i, param in enumerate(params):
                test_params[keys[i]] = param
            setup_and_fit.append([model, X, y, score_func, test_params, f_set, cv_iter])
    
    results = multiprocessor(setup_and_fit_model, setup_and_fit, n_jobs)
    print results
    return(process_results(results, keys))

def feature_selector(model, X, y, test_params, cv_iter, score_func=metrics.auc_score, n_jobs=8):
    print('Starting feature selection on %i features' %len(X[0]))
    sets, setup_and_fit = [], []
    for i in range(len(X[0])):
        #if i < 4: continue  
        for f_set in list(it.combinations(range(len(X[0])),i+1)):
            sets.append(f_set)
    for f_set in sets:
        for test_param in test_params:
            X_en = one_hot_encoding(X[:,f_set])
            setup_and_fit.append([model, X_en, y, score_func, test_param, f_set, cv_iter])
    
    print('Testing %i feature sets' % len(sets))
    results = multiprocessor(setup_and_fit_model, setup_and_fit, n_jobs)
    return(process_results(results, test_params[0].keys()))

def greedy_feature_selector(model, X, y, test_params, cv_iter=5, score_func=metrics.auc_score, seed=36):
    features, num_features, time_start = [], len(X[0]), time.time()
    print('Starting feature selection on %i features' %num_features)
    score_max = 0
    for i in range(num_features): 
        data_sets = []
        test = features + [i]
        #print(test)
        
        X_en = one_hot_encoding(X[:,test])
        for j in range(cv_iter):
            cv = cross_validation.train_test_split(X_en, y, test_size=.20, random_state=seed*j)
            data_sets.append([model, cv, score_func, test_params[0]])        
        results = multiprocessor(model_test, data_sets, cv_iter)
        score = np.mean(results)
        if score > score_max: 
            score_max = score
            features  = test
        print('  auc: %.4f - %i min - %i%% - using features: %s'
              % (score_max, (time.time() - time_start)/60, float(i) / float(num_features) * 100, ', '.join(map(str, features))))
    return(score_max, test_params[0], features)

def greedy_feature_selector_with_hp(model, X_init, y_init, test_params, cv_iter=5, score_func=metrics.auc_score, seed=36):
    features, num_features, time_start, params = [], len(X_init[0]), time.time(), test_params[0]
    cv_init = cross_validation.train_test_split(X_init, y_init, test_size=.20, random_state=seed)
    X, X_test, y, y_test = cv_init
    print('Starting feature selection on %i features' %num_features)
    score_max = 0
    for i in range(num_features): 
        data_sets = []
        test = features + [i]
        
        gd_res = mp_GD_parameterisation(model, X, y, params, c_mod, alphas=[1,10], cv_iter=6, seed=seed*i, f_set=test)
        _, params, _ = gd_res
        
        X_en = one_hot_encoding(X[:,test])
        for j in range(cv_iter):
            cv = cross_validation.train_test_split(X_en, y, test_size=.20, random_state=seed*j)
            data_sets.append([model, cv, score_func, params])        
        results = multiprocessor(model_test, data_sets, cv_iter)
        score = np.mean(results)
        if score > score_max: 
            score_max = score
            features  = test
        print('  auc: %.4f - %i min - %i%% - using features: %s'
              % (score_max, (time.time() - time_start)/60, float(i) / float(num_features) * 100, ', '.join(map(str, features))))
    
    X_train_final, X_test_final = one_hot_encoding(X[:,features], X_test[:,features])
    test_data = [X_train_final, X_test_final, y, y_test]
    final_test_data = [model, test_data, metrics.auc_score, params]
    final_score = model_test(final_test_data)
    return(final_score, params, features)

def feature_builder(X, max_comb, pickle_file=None):
    print('Building features')
    rows, cols = len(X), len(X[0])
    sets = []
    for i in range(cols):
        if i >= max_comb: continue
        for f_set in list(it.combinations(range(cols),i+1)): 
            sets.append(f_set)
    X_new = np.empty((rows,len(sets)))
    for i,f_set in enumerate(sets):
        if len(f_set) == 1: 
            X_new[:,i] = X[:,f_set[0]]
            continue
        X_set = X[:,f_set]
        X_set_en = label_encoder(X_set)
        X_new_col = cantorer(X_set_en)
        X_new_col_en = label_encoder(X_new_col)
        X_new[:,i] = X_new_col_en
    print('Built %i new features, giving %i total' %(len(sets) - cols, len(X_new[0])))
    if pickle_file != None: pickle.dump(X_new, open(pickle_file, 'wb'), protocol=2)
    return(X_new)

def mp_GD_parameterisation(model, X, y, param_dict, modifier, alphas=[1,10], cv_iter=6, seed=32, f_set=[]):
    print('  Starting mp gradient descent parameterisation')
    X_enc = one_hot_encoding(X[:,f_set])
    cv = cross_validation.train_test_split(X_enc, y, test_size=.20, random_state=seed*2)
    X_train, _, y_train, _ = cv
    
    cv_indices = cross_validation.KFold(X_train.shape[0], cv_iter, indices=True, shuffle=True, random_state=seed*3)
    data_sets = []
    for i, indices in enumerate(cv_indices):
        cv_data = [X_train[indices[0]], X_train[indices[1]], y_train[indices[0]], y_train[indices[1]]]
        data_sets.append([model, cv_data[0], cv_data[2], param_dict, modifier, alphas, 3, seed*4*i]) 
    results = multiprocessor(GD_parameterisation, data_sets, len(cv_indices))
    
    scores, alphas = [], []
    for score, alpha in results:
        scores.append(score)
        alphas.append(alpha)
    param_dict = modifier(param_dict, np.mean(alphas))
    
    print('  internal average GD score: %.6f' % np.mean(scores))
    final_test_data = [model, cv, metrics.auc_score, param_dict]
    score = model_test(final_test_data)
    return([score, param_dict, f_set])

def GD_parameterisation(data):
    model, X_, y_, param_dict, modifier, alphas, cv_iter, seed = data
    #print('Starting gradient descent parameterisation')
    score_func  = metrics.auc_score
    #CV indices: is it correct to generate only once, or could this bias the results?
    cv_indices = cross_validation.KFold(X_.shape[0], cv_iter, indices=True, shuffle=True, random_state=seed)

    scores = []
    for alpha in alphas:
        test_param_dict = copy.copy(param_dict)
        test_param_dict = modifier(param_dict, alpha)
        cv_score_sum_init = 0
        for cv_indice in cv_indices:
            sc = sp_cv(model, X_, y_, cv_indice, score_func, test_param_dict)
            cv_score_sum_init += sc
        scores.append(cv_score_sum_init / len(cv_indices))
        
    """http://www.searsmerritt.com/blog/2013/4/machine-learning-with-gradient-descent"""
    a  = 0.05 # can be used to adjust the gradient function
    ep = 0.000005 # the score diff when the function is considered converged
    max_iter = 200

    #print('  Starting GD loop, ep: %f, max_iter %i' %(ep, max_iter))    
    gd_done, t_init = False, time.time()
    while not gd_done:
        grad = (scores[-1] - scores[-2]) / (alphas[-1] - alphas[-2])
        
        d_alpha = grad * a * abs(grad / ep) * alphas[-1]
        #if   abs(alphas[-1] - alphas[-2]) > a * 10: d_alpha = d_alpha * alphas[-1]
        if     alphas[-1] > 30:         d_alpha = d_alpha * alphas[-1]
        elif   alphas[-1] > 10:         d_alpha = d_alpha * alphas[-1] / 2
        if abs(d_alpha) >= alphas[-1]:  d_alpha = alphas[-1] * d_alpha / abs(d_alpha) / 1.2
        alphas.append(alphas[-1] + d_alpha)

        test_param_dict = copy.copy(param_dict)
        test_param_dict = modifier(param_dict, alphas[-1])
        
        cv_score_sum = 0
        for cv_indice in cv_indices:
            cv_score_sum += sp_cv(model, X_, y_, cv_indice, score_func, test_param_dict)
        scores.append(cv_score_sum / len(cv_indices))
        
        d_score = scores[-2] - scores[-1]
        if abs(d_score) < abs(ep): gd_done, cvg,   = True, True
        if len(scores) >= max_iter: gd_done, cvg,  = True, False
        #t_now = (time.time() - t_init)/60
        #report = '  - %.2f min - iter: %i - C: %.4f, grad: %.4f - score: %.6f, diff: %.4f' %(t_now, len(scores), alphas[-1], grad, scores[-1], d_score)
        #print(report)
    
    if not cvg: print('  convergance not reached')
    return(scores[-1], alphas[-1])

def weights_mod(params, val): params['class_weight'] = {0:copy.copy(val), 1:1}; return(params)
    
def c_mod(params, val): params['C'] = val; return(params)

def mp_cv(model, X, y, cv_indices, score_func, params, n_jobs):
    cv_data, data_sets = [], []
    for indices in cv_indices:
        cv_data = [X[indices[0]], X[indices[1]], y[indices[0]], y[indices[1]]]
        data_sets.append([model, cv_data, score_func, params]) 
    results = multiprocessor(model_test, data_sets, n_jobs)
    score = np.mean(results)
    return(score)

def sp_cv(model, X, y, cv_indice, score_func, params):
    cv_data = [X[cv_indice[0]], X[cv_indice[1]], y[cv_indice[0]], y[cv_indice[1]]]
    data  = [model, cv_data, score_func, params]
    score = model_test(data)
    return(score)

# ISSUE! better cv but worse LB score
#
# a. create CV loop outside of hyper_p and feature selector to deal with better CV / worse LB
# b. create hyper parmater loop (gradient decent?) loop to run inside feature selector
#
# 2. try removing (merging) data that occurs < n (~ 1,2,3) times
# 3. try building new features from % occuring with 1 result, counts, etc (see kaggle forums)
#        http://www.kaggle.com/c/amazon-employee-access-challenge/forums/t/5060/using-response-for-new-features
# 4. try creating new features from count of resourse occuring

if __name__ == '__main__':
    directory   = 'D:\Cooper\Google Drive\Kaggle_Amazon\data' #WINDOWS
    #directory   = 'C:\Google Drive\Kaggle_Amazon\data' #WINDOWS 2
    #directory   = '/Users/cooperoelrichs/Google Drive/Kaggle_Amazon/data' #MAC
    training    = 'train_8.csv'
    testing     = 'test_8.csv'
    results     = 'results.csv'
    extended_data = 'extended_data'
    y_train, X_train, index_test, X_test  = get_data(directory, training, testing)
    train_len = len(y_train)
    
    cv_iter, seed, ext_features = 6, 67, 3
    
    for i, (model_name, model) in enumerate(skl_models):
        #for value in greedy_features:
        #f_set = greedy_features[value]
        
        if model_name != 'linsvc': continue
        #if model_name != 'SGD_C_lr' and model_name != 'lr': continue
        #if i > 0: continue
        test_params = paramaters[model_name]
        #f_set = selected_features['lr']
        
        t_now = time.time()
        
        """Create extended features"""
        f_set = greedy_features[12] #[0,1,2,3,4,5,6,7]
        pickle_file = directory + '/' + extended_data + '_%i.p' % ext_features
        #X_extended = feature_builder(np.vstack((X_train, X_test)), ext_features, directory + '/' + extended_data)
        X_extended = pickle.load(open(pickle_file, "rb"))
        X_train_ext, X_test_ext = X_extended[:train_len], X_extended[train_len:]
        
        """Test models"""
        #res  = setup_and_fit_model([model, one_hot_encoding(X_train), y_train, metrics.auc_score, test_params[0], f_set, 6])
        
        print model
        
        """Create new models"""
        #res = mp_GD_parameterisation(model, X_train_ext, y_train, test_params[0], weights_mod, [1,30], 6, seed, f_set)
        res = hyper_parameterisator(model, X_train_ext, y_train, param_grid_dict[model_name], cv_iter, n_jobs=10, f_set=f_set)
        #res = feature_selector(model, X_train, y_train, test_params, cv_iter=cv_iter, score_func=metrics.auc_score, n_jobs=10)
        #res = greedy_feature_selector(model, X_train_ext, y_train, test_params, cv_iter, metrics.auc_score, seed)
        #res = greedy_feature_selector_with_hp(model, X_train_ext, y_train, test_params, cv_iter, metrics.auc_score, seed)
        roc_auc, best_params, best_features = res ; elapsed = time.time() - t_now
        print("AUC (model: %s, t: %.2f min): %f" % (model_name, elapsed/60, roc_auc))
        print("-- Best parameters - %s" % best_params)
        print("-- Best features - %s" % best_features)
    
        create_submission(model, X_train_ext, X_test_ext, y_train, best_params, best_features, directory, 'results_%s_%i.csv' %(model_name, i))
        #create_submission(model, X_train_ext, X_test_ext, y_train, test_params[0], f_set, directory, 'results_%s_%i.csv' %(model_name, i))
        
        
        
        
