import argparse
import os
import time

#importing necessary libraries
import numpy as np
import pandas as pd

import cudf
import cuml
from cuml import RandomForestClassifier as cuRF
from cuml.preprocessing.model_selection import train_test_split
from cuml.metrics.accuracy import accuracy_score

from azureml.core.run import Run
run = Run.get_context()

def main():
    start_script = time.time()
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='location of data')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in RF')
    parser.add_argument('--max_depth', type=int, default=16, help='Max depth of each tree')
    parser.add_argument('--n_bins', type=int, default=8, help='Number of bins used in split point calculation')
    parser.add_argument('--max_features', type=float, default=1.0, help='Number of features for best split')


    args = parser.parse_args()
    data_dir = args.data_dir
    
    print('\n---->>>> cuDF version <<<<----\n', cudf.__version__)
    print('\n---->>>> cuML version <<<<----\n', cuml.__version__)
    
    t1 = time.time()
    df = cudf.read_parquet(os.path.join(data_dir, 'airline_20m.parquet'))
#     df = cudf.read_orc(os.path.join(data_dir, 'airline_20000000.orc'))
    t2 = time.time()
    print('\n---->>>> cuDF time: {:.2f} <<<<----\n'.format(t2-t1))

    X = df[df.columns.difference(['ArrDelay', 'ArrDelayBinary'])]
    y = df['ArrDelayBinary'].astype(np.int32)
    del df
    
    n_estimators = args.n_estimators
    run.log('n_estimators', np.int(args.n_estimators))
    max_depth = args.max_depth
    run.log('max_depth', np.int(args.max_depth))
    n_bins = args.n_bins
    run.log('n_bins', np.int(args.n_bins))
    max_features = args.max_features
    run.log('max_features', np.str(args.max_features))
        
    print('\n---->>>> Training using GPUs <<<<----\n')
    
    # ----------------------------------------------------------------------------------------------------
    # cross-validation folds 
    # ----------------------------------------------------------------------------------------------------
    accuracy_per_fold = []; train_time_per_fold = []; infer_time_per_fold = []; trained_model = [];
    global_best_model = None; global_best_test_accuracy = 0
    
    traintime = time.time()
    # optional cross-validation w/ model_params['n_train_folds'] > 1
    for i_train_fold in range(5):
        print( f"\n CV fold { i_train_fold } of { 5 }\n" )

        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i_train_fold, shuffle = True)

        # train model 
        cu_rf = cuRF(n_estimators=n_estimators, max_depth=max_depth, n_bins=n_bins, max_features=max_features)
        start1 = time.time()
        trained_model = cu_rf.fit(X_train, y_train)
        training_time = time.time() - start1
        train_time_per_fold += [ round( training_time, 4) ]

        # evaluate perf
        start2 = time.time()
        cuml_pred = cu_rf.predict(X_test)
        infer_time = time.time() - start2

        cuml_accuracy = accuracy_score(cuml_pred, y_test) * 100
                
        accuracy_per_fold += [ round( cuml_accuracy, 4) ]
        infer_time_per_fold += [ round( infer_time, 4) ]

        # update best model [ assumes maximization of perf metric ]
        if cuml_accuracy > global_best_test_accuracy :
            global_best_test_accuracy = cuml_accuracy
    
    total_train_inference_time = time.time() - traintime
    run.log('Total training inference time', np.float(total_train_inference_time))
    run.log('Accuracy', np.float(global_best_test_accuracy))
    print( '\n Accuracy             :', global_best_test_accuracy)
    print( '\n accuracy per fold    :', accuracy_per_fold)
    print( '\n train-time per fold  :', train_time_per_fold)
    print( '\n train-time all folds  :', sum(train_time_per_fold))
    print( '\n infer-time per fold  :', infer_time_per_fold)
    print( '\n infer-time all folds  :', sum(infer_time_per_fold))
              
    end_script = time.time()
    print('Total runtime: {:.2f}'.format(end_script-start_script))
    run.log('Total runtime', np.float(end_script-start_script))
    
    print('\n Exiting script')
    

if __name__ == '__main__':
    main()
