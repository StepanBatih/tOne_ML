import sox
sox
import pathlib
import numpy as np
import scipy.io.wavfile as sw
from sklearn import preprocessing

INPUT_DIR_PATH = "D:/ML"
OUTPUT_DIR_PATH_8KHZ = "D:/ML/8KHZ"
OUTPUT_DIR_PATH_22KHZ = "D:/ML/22KHZ"
OUTPUT_DIR_PATH_NORM_8KHZ = "/media/adm_stefan/Data/nedoML/norm_8kHz"
OUTPUT_DIR_PATH_NORM_22KHZ = "/media/adm_stefan/Data/nedoML/norm_22kHz"

iterator = 1
rate_iterator = 1

tfm = sox.Transformer()

output_dir_names = ["/Cats","/Kettles",]
output_dirs = [OUTPUT_DIR_PATH_8KHZ, OUTPUT_DIR_PATH_22KHZ]




for dir in output_dirs:

    for dir_name in output_dir_names:

        for path in pathlib.Path(INPUT_DIR_PATH+dir_name).iterdir():

            if path.is_file():
                
                file_array = sw.read(path)
                    
                np_audio_array = np.array(file_array[1], dtype=float)
                
                if(rate_iterator == 1 or rate_iterator == 3):
                    sample_rate = 8000
                    tfm.rate(sample_rate)
                    
                elif(rate_iterator == 2 or rate_iterator == 4):
                    sample_rate = 22000
                    tfm.rate(sample_rate)
                    
                if(rate_iterator == 3 or rate_iterator == 4):
                    normalized_array = preprocessing.normalize(np_audio_array)
                    tfm.build(input_array=normalized_array, output_filepath=dir+dir_name+"/"+str(iterator)+".wav", sample_rate_in=sample_rate)
                else:  
                    tfm.build(input_array=np_audio_array, output_filepath=dir+dir_name+"/"+str(iterator)+".wav", sample_rate_in=sample_rate)

            iterator += 1

        iterator = 1

    rate_iterator += 1

current_file = open("/media/adm_stefan/Data/nedoML/22kHz/Kettles/1.wav", "r")
file_array = sw.read("/media/adm_stefan/Data/nedoML/22kHz/Kettles/1.wav")                 
np_audio_array = np.array(file_array[1], dtype=float)
normalized_array = preprocessing.normalize(np_audio_array)

print(np_audio_array[1000:1100])
print(normalized_array[1000:1100])
