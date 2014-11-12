import numpy as np

class csvReader(object):
    def get_test(self, fname, index = 0, delimiter = ',', dtype = 'int_', header = 1):
        test = np.loadtxt(fname = fname, dtype = dtype, delimiter = delimiter, skiprows = header)
        X_test = test[:, index + 1:]
        index_test = test[:, index]
        return([index_test, X_test])
    
    def get_train(self, fname, y_index = 0, delimiter = ',', dtype = 'int_', header = 1):
        train = np.loadtxt(fname = fname, dtype = dtype, delimiter = delimiter, skiprows = header)
        X_train = train[:, y_index + 1:]
        y_train = train[:, y_index]
        return([y_train, X_train])