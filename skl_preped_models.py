from sklearn import linear_model, svm
import numpy as np

cluster_model = linear_model.LogisticRegression(C=3)

submission_models = [
                     #['lr', linear_model.LogisticRegression()],
                     #['lr', linear_model.LogisticRegression(C=3.2, class_weight=None)]
                     ['svc_rbf', svm.SVC(kernel='rbf', C=5, probability=True, gamma=0.2, class_weight='auto')],
                     ]

paramaters = {
  'lr'       : [{'C': 10.0009445, 'class_weight' : 'auto'}], #f_set = ['all'] # 3.04082
  #'lr'       : [{'C': 1.763158, 'class_weight' : 'auto'}],
  'svc_lin'  : [{'C': 0.5,              'kernel': 'linear', 'probability' : True, 'class_weight' : 'auto'}],
  'svc_rbf'  : [{'C': 3,  'gamma': 0.3, 'kernel': 'rbf'   , 'probability' : True, 'class_weight' : 'auto'}],
  'svc_poly' : [{'C': 10, 'gamma': 0.1, 'kernel': 'poly'  , 'probability' : True, 'class_weight' : 'auto'}],
  'SGD_C'    : [{'alpha' : 0.000001, 'n_iter' : 500, 'loss' : 'log', 'penalty' : 'l2', 'fit_intercept' : False, 'shuffle' : False, }],
  'SGD_C_lr' : [{'alpha' : 0.000001, 'n_iter' : 500, 'loss' : 'log', 'penalty' : 'l2', 'fit_intercept' : False, 'shuffle' : True, }],
  'linsvc'   : [{'C': 0.5, 'class_weight' : 'auto', 'loss' : 'l2', }],
 }

selected_features = {
                     'lr'       : [0,1,3,4,6,7,8], #[0,1,3,6,7], #[0,1,3,4,5,6],
                     'svc_lin'  : [0,1,3,4,5,6],
                     'svc_rbf'  : [0,1,3,4,6,8],
                     'svc_poly' : [0,1,2,3,5,6,7],
                     'SGD_C'    : [0,1,3,4,5,6,7],
                     }

greedy_features = {
                   1 :[0, 1, 2, 5, 6, 9, 11, 17, 18, 19, 20, 21, 23, 26, 27, 32, 33, 35, 39, 45, 46, 47, 52, 53, 54, 56, 58, 63, 65, 75, 76, 80, 81, 85, 88, 92, 103, 106, 119],
                   2 :[0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18, 20, 21, 23, 26, 27, 33, 36, 39, 45, 46, 47, 52, 53, 54, 55, 56, 58, 63, 65, 73, 75, 76, 78, 80, 81, 84, 85, 88, 92, 103, 106, 119],
                   3 :[0, 1, 2, 3, 5, 6, 7, 17, 18, 20, 21, 23, 27, 32, 33, 36, 39, 45, 46, 47, 50, 52, 53, 54, 58, 63, 65, 75, 76, 80, 81, 85, 88, 92, 100, 103, 110],
                   4 :[0, 1, 2, 3, 5, 6, 7, 17, 18, 20, 21, 23, 27, 32, 33, 36, 39, 45, 46, 47, 50, 52, 53, 54, 58, 63, 65, 75, 76, 77, 80, 81, 85, 88, 92, 97, 100, 103, 110],
                   5 :[0, 1, 4, 5, 6, 7, 9, 11, 17, 18, 19, 20, 21, 22, 23, 26, 27, 32, 33, 39, 45, 46, 47, 50, 52, 53, 54, 58, 60, 65, 75, 76, 78, 80, 81, 85, 88, 92, 97, 103, 106, 107, 109, 113],
                   #6 :[0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 20, 21, 22, 23, 26, 27, 31, 33, 36, 39, 42, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 58, 63, 65, 73, 75, 76, 78, 80, 81, 83, 84, 85, 87, 88, 91, 92, 97, 103, 106, 109, 119, 129, 130, 133, 135, 138, 150, 155, 156, 157, 167, 177, 183, 187, 191, 194, 198, 201, 204, 208, 210, 214, 240, 244, 255, 256, 257, 258, 269, 279, 290, 291, 292, 293, 348, 351, 352, 355, 356, 373, 375, 381, 392, 393, 416, 417, 418, 419, 457, 508, 509, 510],
                   7 :[0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 18, 19, 20, 23, 28, 30, 32, 33, 34, 36, 37, 38, 40, 42, 43, 45, 47, 52, 59, 61, 63, 65, 66, 68, 69, 70, 74, 79, 80, 82, 92, 93, 95, 107, 108, 111, 112, 128, 131, 133, 135, 137, 141, 143, 144, 158, 162, 163, 171, 182, 183, 196, 218],
                   8 :[0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 23, 24, 28, 30, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 45, 47, 52, 59, 61, 63, 65, 66, 68, 69, 70, 74, 79, 80, 82, 84, 91, 92, 93, 94, 95, 97, 107, 108, 111, 128, 131, 135, 137, 141, 143, 144, 162, 163, 164, 166, 167, 182, 183, 184, 196, 207, 208, 211, 212, 216, 218, 219, 225, 233, 234, 235, 244],
                   9 :[0, 1, 5, 6, 7, 15, 16, 17, 18, 19, 20, 23, 24, 25, 27, 28, 30, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 47, 52, 56, 59, 61, 63, 65, 66, 68, 69, 70, 82, 84, 91, 92, 93, 95, 97, 107, 109, 111, 112, 118, 125, 128, 131, 135, 137, 140, 141, 144, 146, 158, 162, 163, 164, 166, 182, 183, 184, 196, 207, 208, 224, 233, 234, 235, 244, 252],
                   10:[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 16, 17, 18, 19, 20, 23, 27, 28, 30, 32, 33, 34, 36, 37, 38, 40, 42, 43, 45, 47, 52, 59, 61, 63, 65, 66, 69, 70, 74, 76, 79, 80, 82, 84],
                   11:[0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 16, 18, 19, 20, 28, 30, 34, 36, 37, 38, 39, 42, 43, 47, 51, 52, 59, 61, 63, 65, 66, 70, 80, 89, 92, 93, 94, 107, 108, 111, 124, 128, 131, 135, 137, 141, 144, 158, 166, 182],
                   12:[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 16, 18, 20, 23, 25, 28, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 48, 52, 59, 63, 66, 68, 70, 74, 79, 80, 82],
                   
                   }

param_grid_dict = {
  'lr'  : [{'C': [3],
            'class_weight' : [{0:i, 1:1} for i in np.linspace(0,40,num=100)],
            'penalty': ['l2'],
            'tol' : [0.00000001]
            }],
  'svc_lin'  : [{'C': np.linspace(1,4,num=40), 'kernel': ['linear'], 'probability' : [True], 
                 'class_weight' : ['auto', None]}],
  'svc_rbf'  : [{'C': np.linspace(1,6,num=5), 'gamma': np.linspace(0.1,0.5,num=5), 'kernel': ['rbf'], 'probability' : [True], 
                 'class_weight' : ['auto']}],
  'svc_poly' : [{'C': np.linspace(1,10,num=5), 'gamma': np.linspace(0.1,0.3,num=2), 'kernel': ['poly'], 'probability' : [True], 
                 'class_weight' : ['auto']}],
  'SGD_C'    : [{'loss' : ['log', 'modified_huber'],
                 'penalty' : ['elasticnet'],
                 'alpha' : np.linspace(0.0000001,0.1,num=20),
                 'l1_ratio' : np.linspace(0,1,num=40),
                 'n_iter' : [50],
                 'shuffle' : [True, False], #[True, False],
                 'random_state' : [324],
                 'fit_intercept' : [False], }],
  'linsvc'   : [{'C': np.linspace(0.01,1,num=40), 'class_weight' : ['auto'], 'loss' : ['l1','l2'], }],
 }

skl_models = [
    #linear_model.ARDRegression(), #sparse mat
    #linear_model.BayesianRidge(), #sparse mat
    #linear_model.ElasticNet(), #predict_proba
    #linear_model.ElasticNetCV(), #predict_proba, precompute is ignored for sparse data
    #linear_model.Lars(), #sparse mat
    #linear_model.LarsCV(), #sparse mat
    #linear_model.Lasso(), #predict_proba
    #linear_model.LassoCV(), #predict_proba
    #linear_model.LassoLars(), #sparse mat
    #linear_model.LassoLarsCV(), #sparse mat
    #linear_model.LassoLarsIC(), #sparse mat
    #linear_model.LinearRegression(), #predict_proba
    ['lr', linear_model.LogisticRegression()], #(C=3),
    #linear_model.MultiTaskLasso(), #sparse mat
    #linear_model.MultiTaskElasticNet(), #sparse mat
    #linear_model.OrthogonalMatchingPursuit(), #sparse mat
    #linear_model.PassiveAggressiveClassifier(), #predict_proba
    #linear_model.PassiveAggressiveRegressor(), #predict_proba
    #linear_model.Perceptron(), #predict_proba
    #linear_model.RandomizedLasso(), #sparse mat
    #linear_model.RandomizedLogisticRegression(), #predict_proba
    #linear_model.Ridge(), #predict_proba
    #linear_model.RidgeClassifier(), #POTENTIAL
    #linear_model.RidgeClassifierCV(), #array is too big.
    #linear_model.RidgeCV(), #array is too big.
    ['SGD_C', linear_model.SGDClassifier()], #(alpha=0.00001, n_iter=100, loss='log')
    ['SGD_C_lr', linear_model.SGDClassifier()],
    #linear_model.SGDRegressor(), #predict_proba
    #linear_model.lars_path(),  #2 args
    #linear_model.lasso_path(),  #2 args
    #linear_model.lasso_stability_path(),  #2 args
    #linear_model.orthogonal_mp(),  #2 args
    #linear_model.orthogonal_mp_gram(),  #2 args
    
    ['svc_lin', svm.SVC()],
    ['svc_rbf', svm.SVC()],
    ['svc_poly', svm.SVC()],
    
    ['linsvc', svm.LinearSVC()],
    
    #['SVC_lin', svm.SVC(kernel='linear', C=1.0, cache_size=600, class_weight=None, coef0=0.0, degree=3, 
    #        gamma=0.0, max_iter=-1, probability=True, shrinking=True, tol=0.001, verbose=False)],
    
    #['SVC_rbf', svm.SVC(kernel='rbf', C=1.0, cache_size=600, class_weight=None, coef0=0.0, degree=3, 
    #        gamma=0.0, max_iter=-1, probability=True, shrinking=True, tol=0.001, verbose=False)],
    
    #['SVC_poly', svm.SVC(kernel='poly', C=1.0, cache_size=600, class_weight=None, coef0=0.0, degree=3, 
    #        gamma=0.0, max_iter=-1, probability=True, shrinking=True, tol=0.001, verbose=False)],
]