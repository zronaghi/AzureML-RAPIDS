import os
import sys
from enum import Enum
import pickle
import numpy as np
import pandas as pd
import cudf
from urllib.request import urlretrieve  
        
def prepare_airline_data(data_dir, nrows):
    if not os.path.exists(data_dir):
        print('creating rf data directory')
        os.makedirs(data_dir)
    
    url = 'http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2'
    
    local_url = os.path.join(data_dir, os.path.basename(url))
    pickle_url = os.path.join(data_dir,'airline'+ ('' if nrows is None else '-' + str(nrows)) + '.pkl')
    
    if os.path.exists(pickle_url):
        return pickle.load(open(pickle_url, 'rb'))
    if not os.path.isfile(local_url):
        urlretrieve(url, local_url)
    
    cols = [
        'Year', 'Month', 'DayofMonth', 'DayofWeek', 'CRSDepTime',
        'CRSArrTime', 'UniqueCarrier', 'FlightNum', 'ActualElapsedTime',
        'Origin', 'Dest', 'Distance', 'Diverted', 'ArrDelay'
    ]
    
    df = cudf.read_csv(local_url, names=cols, nrows=nrows)
    
    print('------Writing to Parquet--------')
    df.to_parquet(data_dir)
              
    print('-------------------')
    print('-Download complete-')
    print('-------------------')
    
    
def upload_airline_data(workspace, datastore, src_dir, tgt_path):
    datastore.upload(
        src_dir=src_dir,
        target_path=tgt_path,
        show_progress=True)
  
    print('-----------------')
    print('-Upload complete-')
    print('-----------------')