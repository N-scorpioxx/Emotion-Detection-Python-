import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
from matplotlib.pyplot import specgram
# import keras
# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense, Embedding
# from keras.layers import LSTM
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from keras.layers import Input, Flatten, Dropout, Activation
# from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
# from keras.models import Model
# from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
# from keras import regularizers
import os
import pandas as pd
import librosa
import glob 
import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np
import sys
import time
from sklearn.utils import shuffle
# from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
 
 
def timeme(method):
    ## Timer function I created for testing 
    def wrapper(*args, **kw):
        startTime = int(round(time.time() * 1000))
        result = method(*args, **kw)
        endTime = int(round(time.time() * 1000))
 
        print(endTime - startTime,'ms')
        return result
 
    return wrapper
 
 
def generateData(source, filename):
    dirList = os.listdir(source)
 
    dataFrame = pd.DataFrame(columns=['features'])
 
    feeling_list = []
    indexCounter = 0

    # print(dirList)
    # return
    for i in dirList:
        soundData, sampleRate = librosa.load(source+i, res_type='kaiser_fast', sr=22050, offset=0.5)
        
        sampleRate = np.array(sampleRate)
        mfccs = np.mean(librosa.feature.mfcc(y=soundData, sr=sampleRate, n_mfcc=13), axis=0)

        features = mfccs
        dataFrame.loc[indexCounter] = [features]
    
        # print(dataFrame)

        feeling_list.append(i)

        print(indexCounter)
        indexCounter += 1

        if (indexCounter > 50):
            break

        # return
 
    labels = pd.DataFrame(feeling_list)
 
    dataFrame = pd.DataFrame(dataFrame['features'].values.tolist())
    dataFrame = pd.concat([dataFrame, labels], axis=1)
    dataFrame = dataFrame.rename(index=str, columns={"0": "label"})
    # dataFrame = shuffle(dataFrame)
    dataFrame = dataFrame.fillna(0)
 
    print(dataFrame)

    # print(dataFrame)
    #  df.to_csv(file_name, sep='\t', encoding='utf-8')
    dataFrame.to_csv('C:\\Users\\M.Naufal\\Desktop\\testing\\hahahah.csv', sep=',', encoding='utf-8')
 
 
# def trainCNNModel(dataset, modelName):
#     dataset = pd.read_csv('data/'+dataset+'.csv', sep='\t', encoding='utf-8', index_col=0)
 
#     ## Quick and Dirty data analysis
#     print(dataset)
 
#     newdf1 = np.random.rand(len(dataset)) < 0.8
#     train = dataset[newdf1]
#     test = dataset[~newdf1]
 
#     # newdf1 = np.random.rand(len(dataset)) < 0.7
#     # train = dataset[newdf1]
#     # testAndCrossValidation = dataset[~newdf1]
 
#     # testSplit = np.random.rand(len(testAndCrossValidation)) < 0.5
#     # test = testAndCrossValidation[testSplit]
#     # crossValidation = testAndCrossValidation[~testSplit]
 
#     trainfeatures = train.iloc[:, :-1]
#     trainlabel = train.iloc[:, -1:]
#     testfeatures = test.iloc[:, :-1]
#     testlabel = test.iloc[:, -1:]
#     # crossValidationfeatures = crossValidation.iloc[:, :-1]
#     # crossValidationlabels = crossValidation.iloc[:, -1:]
 
#     X_train = np.array(trainfeatures)
#     y_train = np.array(trainlabel)
#     X_test = np.array(testfeatures)
#     y_test = np.array(testlabel)
#     # X_crossValidation = np.array(crossValidationfeatures)
#     # y_crossValidation = np.array(crossValidationlabels)
 
#     lb = LabelEncoder()
 
#     y_train = np_utils.to_categorical(lb.fit_transform(y_train))
#     y_test = np_utils.to_categorical(lb.fit_transform(y_test))
#     # y_crossValidation = np_utils.to_categorical(lb.fit_transform(y_crossValidation))
 
#     x_traincnn = np.expand_dims(X_train, axis=2)
#     x_testcnn = np.expand_dims(X_test, axis=2)
#     # x_crossValidationcnn = np.expand_dims(X_crossValidation, axis=2)
 
#     print(x_traincnn.shape)
 
#     model = Sequential()
 
#     model.add(Conv1D(128, 5,padding='same',
#                     input_shape=(x_traincnn.shape[1],1)))
#     model.add(Activation('relu'))
#     model.add(Conv1D(128, 5,padding='same'))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.1))
#     model.add(MaxPooling1D(pool_size=(8)))
#     model.add(Conv1D(128, 5,padding='same',))
#     model.add(Activation('relu'))
#     model.add(Conv1D(128, 5,padding='same',))
#     model.add(Activation('relu'))
#     model.add(Conv1D(128, 5,padding='same',))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#     model.add(Conv1D(128, 5,padding='same',))
#     model.add(Activation('relu'))
#     model.add(Flatten())
#     ## No.of output layers for classification
#     model.add(Dense(2))
#     model.add(Activation('softmax'))
#     opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
 
#     print(model.summary())
 
#     model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#     model.fit(x_traincnn, y_train, batch_size=32, epochs=100, validation_data=(x_testcnn, y_test))
 
#     # print(model.evaluate(X_crossValidation, y_crossValidation))
 
#     '''
#     ## Serialize model to .json file
#     model_json = model.to_json()
#     with open('saved_models/'+modelName+'.json', 'w') as json_file:
#         json_file.write(model_json)
#     # '''
 
#     # '''
#     ## Serialize weights to to HDF5 file
#     model.save_weights('saved_models/'+modelName+'.h5')
#     print('Model saved to disk.')
#     # '''
 
 
@timeme
def main():
    rawDataLocation = 'D:\\CREMA-D\\AudioWAV\\'
    gen = input('Do you want to generate the dataset?\n')
    if gen.lower() == 'y':
        generateData(rawDataLocation, 'Test')
    # trainCNNModel('Test', 'Test')
 
    # generateData(rawDataLocation, 'CREMAMixDataset')
    # trainCNNModel('CREMAMixDataset', 'CREMAMixDataset')
 
 
main()