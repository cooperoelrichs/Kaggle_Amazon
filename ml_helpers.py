from sklearn import (preprocessing)
from csv_reader import csvReader
import numpy as np

def cantor(m,n):
    return(0.5*(m+n)*(m+n+1)+m)

def cantorer(X):
    cols = len(X[0])
    if   cols == 1: return([m                                                                                  for m                 in X])
    elif cols == 2: return([cantor(m,n)                                                                        for m,n               in X])
    elif cols == 3: return([cantor(cantor(m,n),o)                                                              for m,n,o             in X])
    elif cols == 4: return([cantor(cantor(cantor(m,n),o),p)                                                    for m,n,o,p           in X])
    elif cols == 5: return([cantor(cantor(cantor(cantor(m,n),o),p),q)                                          for m,n,o,p,q         in X])
    elif cols == 6: return([cantor(cantor(cantor(cantor(cantor(m,n),o),p),q),r)                                for m,n,o,p,q,r       in X])
    elif cols == 7: return([cantor(cantor(cantor(cantor(cantor(cantor(m,n),o),p),q),r),s)                      for m,n,o,p,q,r,s     in X])
    elif cols == 8: return([cantor(cantor(cantor(cantor(cantor(cantor(cantor(m,n),o),p),q),r),s),t)            for m,n,o,p,q,r,s,t   in X])
    elif cols == 9: return([cantor(cantor(cantor(cantor(cantor(cantor(cantor(cantor(m,n),o),p),q),r),s),t),u)  for m,n,o,p,q,r,s,t,u in X])
    else: raise()
    
def get_best_results(results):
    best_score = 0
    for score, params, f_set in results:
        if score > best_score: best_score, best_params, best_set = score, params, f_set
    return(best_score, best_params, best_set)

def process_results(results, keys):
    #processed_params = ''
    best_score, best_params, best_set = get_best_results(results)
    #for i, key in enumerate(keys):
    #    processed_params += key + ': ' + str(best_params[key])
    #    if i+1 != len(keys): processed_params += ', '
    return([best_score, best_params, best_set])

def get_data(directory, training, testing):
    train_file      = directory + '/' + training
    test_file       = directory + '/' + testing
    y_train, X_train    = csvReader().get_train(fname = train_file)
    index_test, X_test  = csvReader().get_test(fname = test_file)
    return([y_train, X_train, index_test, X_test])

def save_results(predictions, filename, directory):
    """Given a vector of predictions, save results in CSV format."""
    with open(directory + '/' + filename, 'w') as f:
        f.write("id,Action\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))
    print('Results saved to: %s, in: %s' %(filename, directory))
    
def create_submission(model, X_train, X_test, y_train, params, features, directory, file_name='submission.csv'):
    print('Creating Submission')
    model.set_params(**params)
    X_train_en, X_test_en = one_hot_encoding(X_train[:,features], X_test[:,features])
    model.fit(X_train_en, y_train)
    preds = model.predict_proba(X_test_en)[:, 1]
    save_results(preds, file_name, directory)

def label_encoder(X):
    le = preprocessing.LabelEncoder()
    le.fit(X)
    return(le.transform(X))

def one_hot_encoding(X_train, X_test=None):
    encoder = preprocessing.OneHotEncoder()
    if X_test == None: 
        encoder.fit(X_train)
        X_train = encoder.transform(X_train)
        return(X_train)
    else:
        encoder.fit(np.vstack((X_train, X_test)))
        X_train = encoder.transform(X_train)
        X_test  = encoder.transform(X_test)
        return(X_train, X_test)
