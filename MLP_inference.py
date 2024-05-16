from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
import scipy.io.wavfile as wav
import pathlib
import pickle 
import numpy as np
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier

pkl_knn = "pickle_model_mlp.pkl"

with open(pkl_knn, 'rb') as file:
    pickle_model_mlp = pickle.load(file)
    

wav_f="D:/ML/23.05_model_and_presets/4_cat_cut.wav" # 

def inference(wav_file):
    wlen = 0.5
    wstep = 0.5
    arr_x=[]

    rate,sig = wav.read(wav_file)
    mfcc_feat = mfcc(sig,rate,winlen=wlen,winstep=wstep,nfilt=104,nfft=2048,numcep=52)
    temp = arr_x.append(mfcc_feat)
    inf_x = np.concatenate(arr_x)
    
    y_prediction = pickle_model_mlp.predict(inf_x)

    unique, counts = np.unique(y_prediction, return_counts=True)
    sum_pred = sum(counts).astype(float)
    
    result=dict(zip(unique, counts))
    print(result)
    prediction = max(result, key=result.get)
    print(prediction)
    per = (max(counts) / sum_pred) * 100
    print(per)
    
    return (prediction,'{:.2f}'.format(per),'%')
    
print(inference(wav_f))


