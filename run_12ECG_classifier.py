#!/usr/bin/env python

import numpy as np
#import joblib
from get_12ECG_features import get_12ECG_features
from keras.models import Sequential,Model,load_model
from keras.optimizers import Adam
from scipy.io import loadmat
#import numpy as np, os, sys
from keras.layers import Dense, LSTM, Dropout

def run_12ECG_classifier(data,header_data,classes,model):
    threshold = 0.5

    #num_classes = len(classes)
    #current_label = np.zeros(num_classes, dtype=int)
    #current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    data12lead = data
    #features=np.asarray(get_12ECG_features(data,header_data))
    #feats_reshape = features.reshape(1,-1)
    reshaped12lead = data12lead.reshape(data12lead.shape[1],1,12)
    #label = model.predict(reshaped12lead).T
    score = model.predict_proba(reshaped12lead).T
    #current_label[label-2] = 1
    max_pred_val =np.array([score[0].max(),score[1].max(),score[2].max(),score[3].max(),score[4].max(),score[5].max(),score[6].max(),score[7].max(), score[8].max()])

    binary_prediction = []
    for i in range(len(max_pred_val)):
        if (max_pred_val[i] > threshold):
            binary_prediction.append(1)
        elif (max_pred_val[i] < threshold):
            binary_prediction.append(0)
    binary_prediction = np.asarray(binary_prediction)

    #for i in range(num_classes):
    #    current_score[i] = np.array(score[0][i])

    #return current_label, current_score
    return binary_prediction, max_pred_val

def load_12ECG_model():
    # load the model from disk
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, 12)))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='softmax'))

    model.load_weights("LSTM_physionet_comp2020.h5")

    model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['acc'])

    return model
