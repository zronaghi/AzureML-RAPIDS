#Modified from https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/

import argparse
import os
import time

#importing necessary libraries
import numpy as np

import sklearn.svm
from sklearn.datasets.samples_generator import make_gaussian_quantiles
from sklearn.model_selection import train_test_split

import joblib

from azureml.core.run import Run
run = Run.get_context()

def main():
    start_script = time.time()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--kernel', type=str, default='rbf', help='Kernel type to be used in the algorithm')
    parser.add_argument('--C', type=float, default=1.0, help='Regularization parameter')
    
    args = parser.parse_args()
    
    kernel = args.kernel
    run.log('Kernel type', np.str(args.kernel))
    C = args.C
    run.log('Regularization', np.float(args.C))
 
    tol = 1e-3
    gamma = 'scale'
    
    n_samples = 20000
    n_features = 200

    X, y = make_gaussian_quantiles(n_samples=n_samples, n_features=n_features, n_classes=2)

    #dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    print('\n---->>>> Training using CPUs <<<<----\n')
    
    start = time.time()
    sklSVC = sklearn.svm.SVC(kernel=kernel, C=C, tol=tol, gamma=gamma)
    sklSVC.fit(X_train, y_train)
    print('\n---->>>> Training time: {0} <<<<----\n'.format(time.time()-start))
    run.log('Training time', np.float(time.time()-start))
    
    start = time.time()
    skl_pred = sklSVC.predict(X_test)
    print("SKLearn SVC predict time: ", time.time()-start)
    
    skl_accuracy = np.sum(skl_pred==y_test) / y_test.shape[0] * 100
    print("Accuracy: sklSVC {:.2f}%".format(skl_accuracy))
    run.log('Accuracy', np.float(skl_accuracy))
    
#     os.makedirs('outputs', exist_ok=True)
#     # files saved in the "outputs" folder are automatically uploaded into run history
#     joblib.dump(svm_model_linear, 'outputs/model.joblib')
    end_script = time.time()
    run.log('Total runtime', np.float(end_script-start_script))
    
    print('Exiting script')
        

if __name__ == '__main__':
    main()
