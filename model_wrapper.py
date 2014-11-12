import numpy as np
import pylab as pl
import time
from sklearn import (metrics, cross_validation, preprocessing)

class ModelWrapper(object):
    def __init__(self, y_train, X_train, index_test, X_test):
        self.y_train    = y_train
        self.X_train    = X_train
        self.X_test     = X_test
        self.index_test = index_test
        self.y_test     = np.empty(len(X_test))
        self.roc_aucs   = []
        
    def set_model(self, model):
        self.model = model
        
    def one_hot_encoding(self):
        print('One hot encoding')
        """
        OneHotEncoding:
        http://scikit-learn.org/dev/modules/preprocessing.html
        """
        X_train = self.X_train
        X_test  = self.X_test
        
        encoder = preprocessing.OneHotEncoder()
        encoder.fit(np.vstack((X_train, X_test)))
        self.X_train = encoder.transform(X_train)  # Returns a sparse matrix (see numpy.sparse)
        self.X_test  = encoder.transform(X_test)
        #print(self.X_train[:1])
        #print(self.X_test[:1])
        
    def cross_validate_NEW(self, n=10, seed=1):
        #Uses new scikit-learn method - Should be used in scikit-learn 0.14
        X_train = self.X_train
        y_train = self.y_train
        model   = self.model
        
        new_predict = lambda (self, X): self.predict_proba(X)[:,1]
        model.predict = new_predict
        
        cv = cross_validation.cross_val_score(model, X_train, y_train, n_jobs=-1, score_func=metrics.auc_score)
        print(cv)
        return(cv)
    
    def single_cv(self, model, X_train, y_train, seed):
        # if you want to perform feature selection / hyperparameter / optimization, this is where you want to do it
        cv = cross_validation.train_test_split(X_train, y_train, test_size=.20, random_state=seed)
        X_train_cv, X_test_cv, y_train_cv, y_test_cv = cv
        
        model.fit(X_train_cv, y_train_cv) 
        preds = model.predict_proba(X_test_cv)[:, 1]

        fpr, tpr, thresholds = metrics.roc_curve(y_test_cv, preds)
        roc_auc = metrics.auc(fpr, tpr)
        self.fpr = fpr
        self.tpr = tpr
        self.roc_aucs.append(roc_auc)
        
    def cross_validate(self, n=10, seed=1):
        X_train = self.X_train
        y_train = self.y_train
        model   = self.model

        mean_auc = 0.0
        for i in range(n):
            self.single_cv(model, X_train, y_train, seed*n)
        print "AUC (fold %d/%d): %f" % (i + 1, n, np.mean(self.roc_aucs))
            
    def fit_model(self):
        model = self.model
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        
        model.fit(X_train, y_train)
        self.preds = model.predict_proba(X_test)[:, 1]
        
    def save_results(self, file_dir, header):
        preds = self.preds
        
        with open(file_dir, 'w') as f:
            f.write(header + "\n")
            for i, pred in enumerate(preds):
                f.write("%d,%f\n" % (i + 1, pred))
            f.close
    
    def prep_roc_plot(self):
        fpr = self.fpr
        tpr = self.tpr
        roc_auc = self.roc_auc
        
        pl.clf()
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('Receiver operating characteristic example')
        pl.legend(loc="lower right")
        
        self.roc_pl_plot = pl
        
    def plot_roc(self):
        self.roc_pl_plot.show()    
        
            
    ### OLD CODE ------------------------------------------------------------

    def get_duplicates(self, data):
        seen = set()
        seen_add = seen.add
        seen_twice = set(x for x in data if x in seen or seen_add(x))
        return list(seen_twice)
    
class Timer(object):
    def __init__(self, event):
        self.t, self.event = time.time(), event
    def elapsed(self):
        dt = time.time() - self.t
        print(dt)
        return(dt/60)
    def report(self):
        self.dt = time.time() - self.t
        print(' - time since %s: %f' % self.event, self.dt/60)