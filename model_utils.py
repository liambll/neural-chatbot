import numpy as np
from sklearn.cross_validation import train_test_split

'''
 split data into train (70%), test (15%) and valid(15%)
    return (X_train,y_train), (X_valid,y_valid), (X_test,y_test)

'''
def split_dataset(X, y, ratio = [0.7, 0.15, 0.15] ):
    np.random.seed(7)
    X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, test_size=0.3)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size=0.5)
    
    return (X_train,y_train), (X_valid,y_valid), (X_test,y_test)


'''
 generate batches from dataset
     yield (x_gen, y_gen)
'''
def batch_gen(x, y, batch_size):
    if (len(x) % batch_size > 0):
        print("Warning: batch_size does not divide number of records. Some records will not be used.")
    i = 0
    while True:
        if (i+1)*batch_size >= len(x):
            i = 0
        yield x[i*batch_size : (i+1)*batch_size].T, y[i*batch_size : (i+1)*batch_size].T
        i = i+1


    