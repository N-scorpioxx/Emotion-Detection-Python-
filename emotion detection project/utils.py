from __future__ import print_function
from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import time
from math import floor
import librosa
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
import csv
from sklearn import svm
import pickle
from pyAudioAnalysis import audioFeatureExtraction as AFE
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from os import listdir
# from parabolic import parabolic
 
## Option for showing complete arrays in numpy
# np.set_printoptions(threshold=np.nan)
 
 
def voicedData(soundData, sampleRate):
    ## Threshold for voice vs noise
    THRESHOLD = 500
     
    # sampleRate, soundData = wav.read(filename)
 
    totalSamples = soundData.shape[0]
         
    frame = 50 #ms
 
    ## Finding the frame length for sound corresponding to the time frames
    frame = int(sampleRate * (frame/1000))
     
    ## Finding the total number of frames of our sound
    totalFrames = soundData.shape[0]/frame
    framesToCut = int(frame * floor(totalFrames))
     
    ## Padding the overflowing 
    valuesToPad = soundData.shape[0] - framesToCut
    soundData = np.pad(soundData, (0, frame-valuesToPad), 'constant')
    totalFrames = soundData.shape[0]/frame
     
    # tempData = preparePitchFeatureVector(filename)
 
    absSoundData = np.absolute(soundData)
 
    avgAbsSoundData = np.arange(0, totalFrames, 1)
    boolVoicedFrame = np.arange(0, totalFrames, 1)
 
    for i in range(int(totalFrames)):
        avgAbsSoundData[i] = np.mean(absSoundData[0+(i*frame):50+(i*frame)])
        if (avgAbsSoundData[i] > THRESHOLD):
            boolVoicedFrame[i] = 1
        else:
            boolVoicedFrame[i] = 0
        # print(str(i) + " :", avgAbsSoundData[i], " :", boolVoicedFrame[i])
 
 
    # print(boolVoicedFrame)
 
    plt.subplot(3, 1, 1)
    plt.plot(absSoundData)
    plt.subplot(3, 1, 2)
    labels = np.arange(0, 46, 1)
    plt.xticks(labels)
    plt.plot(boolVoicedFrame)
    plt.subplot(3, 1, 3)
    plt.plot(labels, avgAbsSoundData)
    plt.show()
 
    # print(absSoundData.shape)
 
    return boolVoicedFrame
 
 
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
 
 
def pureLabelsGenerator(source, destination):
    # source += "DC/"
    # destination += "DC/"
 
    fileNames = listdir(source)
    # print(fileNames)
 
    VECTOR = np.array([[]])
 
    for file in fileNames:
        print("Working on file no :", file)     
        wavName = file[:-4] + ".npy"
 
        emotion = ""
        for i in wavName:
            if RepresentsInt(i):
                break
            else:
                emotion += i
         
        VECTOR = np.append(VECTOR, emotion)
     
    values = np.array(VECTOR)
    # print(values.shape)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded.shape)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded.shape)
 
    # np.save(destination + "pureLabels.npy", onehot_encoded)
    return onehot_encoded
 
 
def loadModel(source, modelName):
    # print(source+"modelAudioAnalysis.p", end="\n\n\n")
    clf = pickle.load(open(source+modelName+".p", "rb"))
    # clf = pickle.loads(source+"modelAudioAnalysis.p")
    return clf
 
 
def generateModel(source, destination, modelName, option1, option2, MFCC=False, PITCH=False):
    X, y = loaderForAnalysis(source, option1, option2)
 
    if (MFCC):
        x = np.array([])
        for i in range(X.shape[0]):
            x = np.append(x, X[i][10:125])
            # print(x.shape)
 
        X = np.array(np.split(x, X.shape[0]))
     
    if (PITCH):
        x = np.array([])
        for i in range(X.shape[0]):
            x = np.append(x, X[i][0:10])
            # print(x.shape)
 
        X = np.array(np.split(x, X.shape[0]))
 
     
    clf = svm.SVC(kernel='rbf', random_state=0, gamma=0.001, C=6.4)
    # clf = svm.SVC(gamma=0.001, C=10)
    Y = [ np.where(r==1)[0][0] for r in y ]
    Y = np.array(Y)
     
    '''
    # print(X.shape, Y.shape)
    # print(X)
    # print(Y)
    NAN = np.isnan(X)
    # print(NAN.shape)
    counter = 0
    for a in range(NAN.shape[0]):
        for b in range(NAN.shape[1]):
            if (NAN[a][b]):
                print("Error at :", a, b)
                # print("The value :", X[a][b])
                counter += 1
    print("Total NaN values", counter)
    '''
 
    ## 0 = A, 1 = D, 2 = F, 3 = H, 4 = N, 5 = S
 
    ## Thats alnmost 80% of the data
    cutTillTrain = int(X.shape[0]/10.0) * 8
    cutTillValidate = cutTillTrain + int(X.shape[0]/10.0) * 1
    cutTillTest = X.shape[0]
 
    clf.fit(X[0:cutTillTrain], Y[0:cutTillTrain])
    print("Samples from " + str(0) + " till " + str(cutTillTrain))
    pickle.dump(clf, open(destination+modelName+".p", "wb"))
 
 
def loaderForAnalysis(source, option1="", option2=""):
    features = np.load(source+option1+option2+'features.npy')
    labels = np.load(source+option1+option2+'pureLabels.npy')
 
    # print(features.shape)
    # print(labels.shape)
 
    return features, labels
 
 
def loaderForNpyFiles(source, destination, listOfFiles):
    # tempSource = []
    # for i in listOfFiles:
    #   tempSource.append(source+i+"/")
    # source = tempSource
 
    VECTOR = np.array([[]])
 
    # for s in source:
    fileNames = listdir(source)
    print(fileNames)
    for file in fileNames:
        print("Working on file no :", file)     
        wavName = file[:-4] + ".npy"
         
        VECTOR = np.append(VECTOR, np.load(source+wavName))
         
    ## 125 -> number of features per data element
    goodFiles = VECTOR.shape[0]/122
    # print(goodFiles)
    VECTOR = np.array(np.split(VECTOR, goodFiles))
    LABELS = pureLabelsGenerator(source, destination)
     
    ## Dataset shuffler
    idx = np.random.permutation(len(VECTOR))
    VECTOR, LABELS = VECTOR[idx], LABELS[idx]
 
    np.save(destination + "features.npy", VECTOR)
    np.save(destination + "pureLabels.npy", LABELS)
     
 
def extractorToNpy(source, destination, listOfFiles):
    tempSource = []
    for i in listOfFiles:
        tempSource.append(source+i+"/")
    source = tempSource
 
    counter = 0
    for s in source:
        fileNames = listdir(s)
        for file in fileNames:
            print("Working on file no :", file)     
            wavName = file[:-4] + s[-3:-1] + ".npy"
 
            VECTOR = np.array([])
            VECTOR = np.append(VECTOR, preparePitchFeatureVector(s+file))
            VECTOR = np.append(VECTOR, prepareMfccFeatureVector(s+file))
            np.save(destination + wavName, VECTOR)
            break
        break
 
 
def timeme(method):
    ## Timer function I created for testing 
    def wrapper(*args, **kw):
        startTime = int(round(time.time() * 1000))
        result = method(*args, **kw)
        endTime = int(round(time.time() * 1000))
 
        print(endTime - startTime,'ms')
        return result
 
    return wrapper
 
 
def statMfcc(mfccData):
    ## Statistics related to MFCCs
    DATA = np.array([])
    DATA = np.append(DATA, np.mean(mfccData, axis=1))
    DATA = np.append(DATA, np.var(mfccData, axis=1))
    DATA = np.append(DATA, np.amin(mfccData, axis=1))
    DATA = np.append(DATA, np.amax(mfccData, axis=1))
     
    # print(DATA.shape)
     
    return DATA
 
 
def statMfccD(derivativeOfMfccData):
    ## Statistics related to derivative MFCCs 
    DATA = np.array([])
    DATA = np.append(DATA, np.mean(derivativeOfMfccData, axis=1))
    DATA = np.append(DATA, np.var(derivativeOfMfccData, axis=1))
    DATA = np.append(DATA, np.amin(derivativeOfMfccData, axis=1))
    DATA = np.append(DATA, np.amax(derivativeOfMfccData, axis=1))
     
    # print(DATA.shape)
 
    return DATA
 
 
def statMfccO(statMfccData):
    ## Statistics related to overall MFCCs
    DATA = np.array([])
    DATA = np.append(DATA, np.mean(statMfccData[0:13])) 
    DATA = np.append(DATA, np.var(statMfccData[13:26]))
    DATA = np.append(DATA, np.min(statMfccData[26:39]))
    DATA = np.append(DATA, np.max(statMfccData[39:52]))
     
    # print(DATA.shape)
 
    return DATA
 
 
def statMfccOD(derivativeStatMfccData):
    ## Statistics related to overall derivative MFCCs
    DATA = np.array([])
    DATA = np.append(DATA,np.mean(derivativeStatMfccData[0:13]))
    DATA = np.append(DATA,np.var(derivativeStatMfccData[13:26]))
    DATA = np.append(DATA,np.min(derivativeStatMfccData[26:39]))
    DATA = np.append(DATA,np.max(derivativeStatMfccData[39:52]))
 
    # print(DATA.shape)
 
    return DATA
 
 
def prepareMfccFeatureVector(filename):
    ## Loading the sound file
    sampleRate, soundData = wav.read(filename)
    ## Features for the first 13 MFCCs 
    melFrame = 35
     
    ## Converting sound to floating points 
    soundData = soundData.astype("float32")
 
    ## Producing mfcc for one extra frame
    mfccData = librosa.feature.mfcc(y=soundData, sr=sampleRate, hop_length=melFrame, n_mfcc=13)
    mfccData = ((mfccData.T)[:-1]).T
    # print(mfccData)
    derivativeOfMfccData = np.array([])
     
    counter = 0
    for i in mfccData:
        if counter == 0:
            delta = i
            derivativeOfMfccData = np.append(derivativeOfMfccData, delta)
        else:
            delta = i - prev
            derivativeOfMfccData = np.append(derivativeOfMfccData, delta)
        prev = i
        counter+=1
 
    derivativeOfMfccData = np.array(np.split(derivativeOfMfccData, mfccData.shape[0]))
 
    # derivativeOfMfccData = np.apply_along_axis(gradientOfMfcc, 1, mfccData)
     
    mfccFeatureVector = np.array([])
    mfccFeatureVector = np.append(mfccFeatureVector, statMfcc(mfccData))
    mfccFeatureVector = np.append(mfccFeatureVector, statMfccD(derivativeOfMfccData))
    mfccFeatureVector = np.append(mfccFeatureVector, statMfccO(mfccFeatureVector[0:52]))
    mfccFeatureVector = np.append(mfccFeatureVector, statMfccOD(mfccFeatureVector[52:104]))
 
    # print("mfccData : " + str(mfccData.shape))
    # print("derivativeOfMfccData : " + str(derivativeOfMfccData.shape))
    # print(mfccFeatureVector.shape)
 
    return mfccFeatureVector
 
 
np.set_printoptions(threshold=np.nan)
def arraySlicer(data, slices, overlap):
    distance = overlap
 
    # print(slices*distance, data.shape[0])
    # if (slices*distance > data.shape[0]):
    #   print("Slicing not possible")
    # else:
    data = np.array(zip(*(data[i*distance:] for i in range(slices))))
    return data.T
 
 
# def freq_from_fft(frame, sampleRate):
#     """
#     Estimate frequency from peak of FFT
#     """
#     # Compute Fourier transform of windowed signal
#     windowed = frame * blackmanharris(len(frame))
#     f = np.fft.rfft(windowed)
 
#     # Find the peak and interpolate to get a more accurate peak
#     i = np.argmax(abs(f))  # Just use this for less-accurate, naive version
#     true_i = parabolic(np.log(abs(f)), i)[0]
 
#     # Convert to equivalent frequency
#     return sampleRate * true_i / len(windowed)
 
 
def getPitchesForEachFrame(soundData, sampleRate):
    frameLength = np.floor(soundData.shape[0]/25)
    frameOverLap = np.floor(soundData.shape[0]/10)
    print(frameLength)
    soundData = arraySlicer(soundData, frameLength, frameOverLap)
    print(soundData.shape)
    # print(soundData[1])
 
 
def preparePitchFeatureVector(filename):
    ## Find a better way to get pitches for our audio clip
    ## Function Prototype : pysptk.sptk.rapt(x, fs, hopsize, min=60, max=240, voice_bias=0.0, otype='f0')
    ## The current is removing zeroed (unvoiced) pitches
    ## Still have to understand voiced and unvoiced data from audio
    # pitches = pysptk.sptk.rapt(x=soundData, fs=sampleRate, hopsize=totalFrames, min=60, max=250, voice_bias=0.5, otype="f0")
    # pitches = pitches[np.nonzero(pitches)]
 
    '''-----------------------------------------------------PITCH FEATURES-----------------------------------------------------''' 
    ## Example : pitch = pYAAPT.yaapt(signal, **{'f0_min' : 150.0, 'frame_length' : 15.0, 'frame_space' : 5.0})
    signal = basic.SignalObj(filename)
    pitch = pYAAPT.yaapt(signal, **{'f0_min' : 60.0, 'f0_max' : 360, 'frame_length' : 25.0, 'frame_space' : 10.0})
    pitches = pitch.samp_values
 
    testSampleRate, testSoundData = wav.read(filename)
 
    getPitchesForEachFrame(testSoundData, testSampleRate)
 
    ## Getting the voiced part of the sound clip
    boolVoiced = np.array([])
 
    for i in range(len(pitches)):
        if (pitches[i] == 0):
            boolVoiced = np.append(boolVoiced, 0)
        else:
            boolVoiced = np.append(boolVoiced, 1)
 
    derivativeOfPitches = np.array([])
 
    counter = 0
    for i in pitches:
        if counter == 0:
            delta = i
            derivativeOfPitches = np.append(derivativeOfPitches, delta)
        else:
            delta = i - prev
            derivativeOfPitches = np.append(derivativeOfPitches, delta)
        prev = i
        counter+=1
 
    derivativeOfPitches = np.array(np.split(derivativeOfPitches, pitches.shape[0]))
     
    # print(boolVoiced)
 
    # sampleRate, soundData = wav.read(filename)
    # plt.subplot(2, 2, 1)
    # plt.plot(soundData)
    # plt.subplot(2, 2, 3)
    # plt.plot(pitch.samp_values)
    # plt.subplot(2, 2, 4)
    # plt.plot(boolVoiced)
    # plt.subplot(2, 2, 2)
    # plt.plot(voicedData(filename))    
    # plt.show()
    # print("Pitches frames", pitches.shape)
 
 
    pitchFeatureVector = np.array([])
 
    ## soundData statistics
    ## Features 0 to 4
    pitchFeatureVector = np.append(pitchFeatureVector, np.mean(pitches))
    pitchFeatureVector = np.append(pitchFeatureVector, np.median(pitches))
    pitchFeatureVector = np.append(pitchFeatureVector, np.max(pitches[np.nonzero(pitches)]))
    pitchFeatureVector = np.append(pitchFeatureVector, np.min(pitches[np.nonzero(pitches)]))
    pitchFeatureVector = np.append(pitchFeatureVector, np.var(pitches))
 
    ## soundData Derivative statistics
    ## Features 5 to 9
    pitchFeatureVector = np.append(pitchFeatureVector, np.mean(derivativeOfPitches))
    pitchFeatureVector = np.append(pitchFeatureVector, np.median(derivativeOfPitches))
    pitchFeatureVector = np.append(pitchFeatureVector, np.max(derivativeOfPitches[np.nonzero(derivativeOfPitches)]))
    pitchFeatureVector = np.append(pitchFeatureVector, np.min(derivativeOfPitches[np.nonzero(derivativeOfPitches)]))
    pitchFeatureVector = np.append(pitchFeatureVector, np.var(derivativeOfPitches))
 
    # print(pitchFeatureVector.shape)
    '''-----------------------------------------------------PITCH FEATURES-----------------------------------------------------'''
 
    '''-----------------------------------------------------AVERAGE ENERGIES-----------------------------------------------------'''
    sr, sd = wav.read(filename)
    # boolVoiced = voicedData(sd, sr)
 
    frame = 50 #ms
 
    ## Finding the frame length for sound corresponding to the time frames
    frame = int(sr * (frame/1000))
     
    ## Finding the total number of frames of our sound
    totalFrames = sd.shape[0]/frame
    framesToCut = int(frame * floor(totalFrames))
     
    ## Padding the overflowing 
    valuesToPad = sd.shape[0] - framesToCut
    soundData = np.pad(sd, (0, frame-valuesToPad), 'constant')
    totalFrames = soundData.shape[0]/frame
 
    soundData = np.array(np.split(soundData, totalFrames))
    # soundData = soundData[:-1]
 
    ## Average energies of soundData
    ## Have to ratio it according to voiced and unvoiced data
    ## Function prototype :  librosa.feature.rmse(y=None, S=None, frame_length=2048, hop_length=512, center=True, pad_mode='reflect)
    # averageEnergies = np.mean(librosa.feature.rmse(y=soundData, hop_length=int(totalFrames), center=True, pad_mode='reflect').T, axis=0)[0]
    voicedEnergies = np.array([])
    unvoicedEnergies = np.array([])
    '''
    for i in range(boolVoiced.shape[0]):
        if (boolVoiced[i] == 0):
            unvoicedEnergies = np.append(unvoicedEnergies, AFE.stEnergy(soundData[i]))
        else:
            voicedEnergies = np.append(voicedEnergies, AFE.stEnergy(soundData[i]))
 
    ## Feature 10
    # print("voicedEnergies", voicedEnergies)
    voicedEnergies = np.mean(voicedEnergies)
    pitchFeatureVector = np.append(pitchFeatureVector, voicedEnergies)
    ## Feature 11
    unvoicedEnergies = np.mean(unvoicedEnergies)
    # print("unvoicedEnergies", voicedEnergies)
    pitchFeatureVector = np.append(pitchFeatureVector, unvoicedEnergies)
    ## Checking for NaN values
    if (np.isnan(pitchFeatureVector[11])):
        pitchFeatureVector[11] = 0
 
    #-----------------------------------------------------AVERAGE ENERGIES-----------------------------------------------------
 
    #------------------------------------------------------SPEAKING RATE------------------------------------------------------
    ## Speaking rate of soundData (inverse of the average length of the voiced part of an utterance)
    voicedParts = np.array([])
    # print(boolVoiced)
 
    LENGTH = 0
    for i in range(boolVoiced.shape[0]):
        if (boolVoiced[i] == 1):
            LENGTH += 1
        elif (LENGTH > 0 and boolVoiced[i] == 0):
            ## 50 because thats the frame length we are going with.
            voicedParts = np.append(voicedParts, LENGTH*(50/1000))
            LENGTH = 0
     
    if (LENGTH != 0):
        voicedParts = np.append(voicedParts, LENGTH*(50/1000))
        LENGTH = 0
 
 
    # print(voicedParts)
    ## Speaking rate made out to be in words per second
    # print(boolVoiced)     
    speakingRate = 1 / (np.mean(voicedParts))
    # print("Speaking rate :", speakingRate)
    ## Feature 12
    pitchFeatureVector = np.append(pitchFeatureVector, speakingRate)
    #------------------------------------------------------SPEAKING RATE------------------------------------------------------
    '''
 
    return pitchFeatureVector