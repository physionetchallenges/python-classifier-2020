#!/usr/bin/env python

import numpy as np, os, sys
import joblib
from get_12ECG_features import get_12ECG_features

def run_12ECG_classifier(data,header_data,loaded_model):


    # Use your classifier here to obtain a label and score for each class.
    model = loaded_model['model']
    imputer = loaded_model['imputer']
    classes = loaded_model['classes']

    features=np.asarray(get_12ECG_features(data,header_data))
    feats_reshape = features.reshape(1, -1)
    feats_reshape = imputer.transform(feats_reshape)
    current_label = model.predict(feats_reshape)[0]
    current_label=current_label.astype(int)
    current_score = model.predict_proba(feats_reshape)
    current_score=np.asarray(current_score)
    current_score=current_score[:,0,1]

    return current_label, current_score,classes

def load_12ECG_model(input_directory):
    # load the model from disk 
    f_out='finalized_model.sav'
    filename = os.path.join(input_directory,f_out)

    loaded_model = joblib.load(filename)

    return loaded_model
