# CNN Architecture is constructed in create_cnn_model() function:
    # 3 series of Convolution - Batch Normalization - Convolution - MaxPooling - Batch Normalization, followed by 2 fully connected layers, and then output layer
    # Regularization using Dropout and L2 Norm (only on fully connected layers)
# Ensemble model: train many CNNs and ensemble predictions of some best CNNs
# Single CNN has val_acc ~ 76%. Ensemble model has val_acc ~ 84%

import numpy as np
import pickle
np.random.seed(7)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

import keras
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')

## TRAIN, VALIDATION SET  
# X has dimension 32x32x3 and is already normalized
with open('train_feat.pickle','rb') as f:
    X_train_label = pickle.load(f)
X_train = X_train_label.astype('float32')


with open('validation_feat.pickle','rb') as f:
    X_valid_label = pickle.load(f)
X_valid = X_valid_label.astype('float32')

#Y has 10 classes and need one-hot encoding
with open('train_lab.pickle','rb') as f:
    Y_train_label = pickle.load(f)
Y_train = np_utils.to_categorical(Y_train_label, 10)

with open('validation_lab.pickle','rb') as f:
    Y_valid_label = pickle.load(f)
Y_valid = np_utils.to_categorical(Y_valid_label, 10)

# Parameter
num_classes = Y_train.shape[1]
batch_size = 64
epochs = 100
no_estimators = 50

# create CNN
def create_cnn_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3), activation='relu', border_mode='same'))
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', W_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', W_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

"""           
## BASIC DEEP NETWORK MODEL ~ val_err 24%
model = create_cnn_model()
#model.load_weights('model_cnn.h5')

# monitor val_loss and terminate training if no improvement
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, \
            patience=30, verbose=2, mode='auto')
# save best model based on val_acc during training
checkpoint = keras.callbacks.ModelCheckpoint('model_cnn.h5', monitor='val_acc', \
            verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
# enable visualize training with command tensorboard --logdir=/tmp
tensorboard = keras.callbacks.TensorBoard(log_dir='/tmp')

model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), nb_epoch=epochs, \
                batch_size=batch_size, verbose=2, callbacks=[checkpoint, tensorboard])
# Final evaluation of the model
scores = model.evaluate(X_valid, Y_valid, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
"""

## ENSEMBLE MODELS
# Train many CNN models
for i in np.arange(no_estimators):
    # check point to save best model
    checkpoint = keras.callbacks.ModelCheckpoint('models/model_'+str(i)+'.h5', monitor='val_acc', \
            verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
    # clear GPU memory
    model = None
    K.clear_session()
    
    # train model
    model = create_cnn_model()
    model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), nb_epoch=100, \
                batch_size=64, verbose=2, callbacks=[checkpoint])

# Check individual scores - val_acc and get list of the best models
score_list = []
model = create_cnn_model()
for i in np.arange(no_estimators):    
    model.load_weights('models/model_'+str(i)+'.h5')
    scores = model.evaluate(X_valid, Y_valid, verbose=0)
    score_list.append(scores[1])

score_list = np.array(score_list)
score_list_filter = score_list >= 0.76 # threshold to select best model

"""   
# Ensemble predictions ~ val_err 16%
predictions = np.zeros((Y_valid.shape[0],Y_valid.shape[1]))
model = create_cnn_model()
for i in np.arange(no_estimators): 
    if score_list_filter[i] == False:
        continue
    model.load_weights('models/model_'+str(i)+'.h5')
    prediction = model.predict_classes(X_valid, batch_size=200, verbose=0)
    prediction_onehot = np_utils.to_categorical(prediction, 10)
    predictions = predictions + prediction_onehot*score_list[i]

predictions_label = np.argmax(predictions, axis=1)
evaluate = np.equal(predictions_label, Y_valid_label)
print("Ensemble Error: %.2f%%" % (100-np.mean(evaluate)*100))
"""

## PERFORM PREDICTION ON TEST SET
with open('test_feat.pickle','rb') as f:
    X_test_label = pickle.load(f)
X_test = X_test_label.astype('float32')

predictions = np.zeros((Y_valid.shape[0],Y_valid.shape[1]))
model = create_cnn_model()
for i in np.arange(no_estimators): 
    if score_list_filter[i] == False: # skip bad model
        continue
    model.load_weights('models/model_'+str(i)+'.h5')
    prediction = model.predict_classes(X_test, batch_size=200, verbose=0)
    prediction_onehot = np_utils.to_categorical(prediction, 10)
    predictions = predictions + prediction_onehot*score_list[i]

predictions_label = np.argmax(predictions, axis=1)
output = predictions_label.tolist()

with open('test_lab.pickle','wb') as f:
    pickle.dump(output, f)

