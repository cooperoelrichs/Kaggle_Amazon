from sklearn import (metrics, cross_validation, preprocessing)
from csv_reader import csvReader
from model_wrapper import ModelWrapper
from skl_preped_models import (skl_models, cluster_model)
from clusterer import Clusterer
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
    X_train, X_test = one_hot_encoding(X_train, X_test)
    return([y_train, X_train, index_test, X_test])

if __name__ == '__main__':

    directory   = 'D:\Cooper\Google Drive\Kaggle_Amazon\data' #WINDOWS
    #directory   = '/Users/cooperoelrichs/Google Drive/Kaggle_Amazon/data' #MAC
    training    = 'train.csv'
    testing     = 'test.csv'
    results     = 'results.csv'
    
    y_train, X_train, index_test, X_test  = get_data(directory, training, testing)
    
    models, data, seed = [], [], 13
    for i,model in enumerate(skl_models):
        for j in range(4): data.append([model, X_train, y_train, j*seed])
        
        pool = Pool()
        rocs = pool.map(single_cv, data)
        print "AUC (model %d/%d): %f" % (i + 1, len(skl_models), np.mean(rocs))
        
        #if i > 1: continue
        #print(model)
        #processor = ModelWrapper(y_train, X_train, index_test, X_test)
        #processor.set_model(model)
        #processor.one_hot_encoding()
        #processor.cross_validate(n=6, seed=1)
        
        #processor.fit_model()
        #processor.save_results(file_dir=results_file, header="id,ACTION")
        #processor.prep_roc_plot()
        #models.append(processor)
        
    #cluster = Clusterer(models)
    #cluster.data(y_train, X_train, index_test,  X_test)
    #cluster.cluster_model(cluster_model)
    #cluster.cross_validate(seed=12, iters=1)
    #cluster.prep_roc_plot()
    #cluster.plot_roc()


