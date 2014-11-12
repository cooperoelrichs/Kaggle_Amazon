import numpy as np
import pylab as pl
from sklearn import (metrics, cross_validation, preprocessing)

class Clusterer(object):
    def __init__(self, models):
        print('NOTE: Class must be updated, modify to use sklearn.pipeline')
        self.models = models
        self.how_many_models()
        
    def cluster_model(self, cluster_model):
        self.cluster_model = cluster_model
        
    def how_many_models(self):
        print('Cluster of %d models' % len(self.models))
        
    def data(self, y_train, X_train, index_test,  X_test):
        self.y_train    = y_train
        self.X_train    = X_train
        self.X_test     = X_test
        self.index_test = index_test
        self.y_test     = np.empty(len(X_test))
        
    def one_hot_encoding(self, X_train, X_test):
        print('One hot encoding')
        """
        OneHotEncoding:
        http://scikit-learn.org/dev/modules/preprocessing.html
        """
        encoder = preprocessing.OneHotEncoder()
        encoder.fit(np.vstack((X_train, X_test)))
        X_train = encoder.transform(X_train)  # Returns a sparse matrix (see numpy.sparse)
        X_test  = encoder.transform(X_test)
        return(X_train, X_test)

    def cross_validate(self, iters=10, seed=1):
        print('cross validating')
        models  = self.models
        cluster_model = self.cluster_model
        X_train = self.X_train
        y_train = self.y_train
        
        self.mean_auc = 0.0
        for i in range(iters):
            cv = cross_validation.train_test_split(X_train, y_train, test_size=.20, random_state=seed*i)
            X_train_cv, X_test_cv, y_train_cv, y_test_cv = cv
            X_train_one_hot, X_test_one_hot = self.one_hot_encoding(X_train_cv, X_test_cv)

            X_train_cluster = np.empty((len(X_train_cv), len(models)))
            X_test_cluster  = np.empty((len(X_test_cv), len(models)))
            
            for j, model_wrapper in enumerate(models):
                
                model_wrapper.model.fit(X_train_one_hot, y_train_cv)
                train_preds = model_wrapper.model.predict_proba(X_train_one_hot)[:, 1]
                test_preds  = model_wrapper.model.predict_proba(X_test_one_hot)[:, 1]
                
                self.run_metrics(y_test_cv, test_preds)
                print "model %d: AUC (fold %d/%d): %f" % (j, i + 1, iters, self.roc_auc)

                for n, x in enumerate(train_preds): X_train_cluster[n, j] = x
                for n, x in enumerate(test_preds):  X_test_cluster[n, j]  = x

            print(X_train_cluster[:9])
            
            cluster_model.fit(X_train_cluster, y_train_cv)
            cluster_preds = cluster_model.predict_proba(X_test_cluster)[:, 1]
            self.run_metrics(y_test_cv, cluster_preds)
            if i+1 == iters: print "cluster AUC (fold %d/%d): %f" % (i + 1, iters, self.roc_auc)

    def run_metrics(self, y_test_cv, cluster_preds):
        fpr, tpr, thresholds = metrics.roc_curve(y_test_cv, cluster_preds)
        roc_auc = metrics.auc(fpr, tpr)
        self.mean_auc += roc_auc
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc

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