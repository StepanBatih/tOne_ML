from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
import scipy.io.wavfile as wav
import pathlib
import pickle 
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier , KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

wlen = 0.1
wstep = 0.1

OUTPUT_DIR_PATH_8KHZ = "D:/ML/8KHZ"
#OUTPUT_DIR_PATH_22KHZ = "D:/ML/22KHZ"

iterator = 1
rate_iterator = 1

cats_8kHz=[]

kettles_8kHz=[]

slience_8kHz=[]

cats_22kHz=[]
kettles_22kHz=[]
slience_22kHz=[]



output_dir_names = ["/Cats","/Kettles","/Silence"]
output_dirs = [OUTPUT_DIR_PATH_8KHZ]

for dir in output_dirs:
    #print(dir)
    for dir_name in output_dir_names:
        #print(dir_name)
        for path in pathlib.Path(dir+dir_name).glob('**/*.wav'):
            print(path)
            rate,sig = wav.read(path)
            if (iterator == 1):
                mfcc_feat = mfcc(sig,rate,winlen=wlen,winstep=wstep,nfilt=52,nfft=2048,numcep=26)
                cats_8kHz.append(mfcc_feat)
            elif (iterator == 2):
                mfcc_feat = mfcc(sig,rate,winlen=wlen,winstep=wstep,nfilt=52,nfft=2048,numcep=26)
                kettles_8kHz.append(mfcc_feat)
            elif (iterator == 3):
                mfcc_feat = mfcc(sig,rate,winlen=wlen,winstep=wstep,numcep=26)
                slience_8kHz.append(mfcc_feat)
            elif (iterator == 4):
                mfcc_feat = mfcc(sig,rate,winlen=wlen,winstep=wstep)
                mfcc_array[3].append(mfcc_feat)
            elif (iterator == 5):
                mfcc_feat = mfcc(sig,rate,winlen=wlen,winstep=wstep)
                mfcc_array[4].append(mfcc_feat)
            elif (iterator == 6):
                mfcc_feat = mfcc(sig,rate,winlen=wlen,winstep=wstep)
                mfcc_array[5].append(mfcc_feat)
           
 
        iterator += 1

cat_x = np.concatenate(cats_8kHz)
slience_x = np.concatenate(slience_8kHz)
kettles_x = np.concatenate(kettles_8kHz)

X = np.concatenate([cat_x, slience_x,kettles_x])

cat_y = ["cat" for _ in range(cat_x.shape[0])]
slience_y = ["slience" for _ in range(slience_x.shape[0])]
kettles_y = ["kettles" for _ in range(kettles_x.shape[0])]

y = np.concatenate([cat_y,slience_y,kettles_y])










