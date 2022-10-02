import librosa
import os
import pickle
import numpy as np
import speech_recognition as sr
from urllib import response
from flask import Flask, request
from flask import jsonify
from sklearn import model_selection
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import nltk
import xgboost as xgb
from LeXmo import LeXmo
nltk.download('omw-1.4')
nltk.download('punkt')

emotionString = [ "Fear", "Happy", "Neutral", "Sad", "Anger", "Disgust" ]
pathToModel = "./pkl/combinedModel.pkl"
fileName = "./pkl/speechModel.pkl"
combinedModel = pickle.load(open(pathToModel, 'rb'))
speechModel = pickle.load(open(fileName, 'rb'))

def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc
    
def convertToText(path):
    """takes in a path to audio files and returns interpreted text"""
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audioData = r.record(source)
        text = r.recognize_google(audioData)
        return text

def convertToFormat(inp):
    out = np.array([])
    for i in inp:
        temp = np.array([i])
        out = np.append(out, temp)
    return np.array([out])

def prepCombinedModel(pred, em): ## em is a dict and pred is a 2d list
    return {
        "t-happy" : em['joy'] + (0.5 * em['surprise']),
        "t-angry" : em['anger'],
        "t-fear" : em['fear'] + (0.5 * em['surprise']),
        "t-sad" : em['sadness'],
        # "t-surprise" : em['surprise'],
        "t-disgust" : em['disgust'],
        # "t-neutral" : (1 - (em['positive']-em['negative'])),
        # "t-neutral" : 0.002,
        "s-happy" : pred[0][6] + (0.5 * pred[0][5]),
        # "s-surprise" : pred[0][5],
        "s-sad" : pred[0][4],
        # "s-neutral" : pred[0][3],
        "s-disgust" : pred[0][2],
        "s-angry" : pred[0][1],
        "s-fear" : pred[0][0] + (0.5 * pred[0][5])
    }

def processAudio(path):
    inpForSpeech = extract_mfcc(path)
    formattedInput = convertToFormat(inpForSpeech)    
    prediction = speechModel.predict(formattedInput)
    inputTextModel = convertToText(path)
    print(inputTextModel)
    emo = LeXmo.LeXmo(inputTextModel) 
    standardized = prepCombinedModel(prediction, emo)
    inpToCombinedModel = [[ x for x in standardized.values() ]]
    inpToCombinedModel = np.array(inpToCombinedModel)
    matrix = xgb.DMatrix(inpToCombinedModel)
    ourPrediction = combinedModel.predict(matrix)
    return emotionString[int(ourPrediction[0])]
    

app = Flask(__name__)
@app.route('/', methods = ['GET', 'POST'])
def login():
    if request.method == 'GET':
        return jsonify({"message" : "howareyou"})
    elif request.method == 'POST':
        studentName = request.form['studentName']
        return studentName + " is a MotherFucker"

@app.route('/audio', methods = ['POST'])
def processAudioEndpoint():
    if request.method == 'POST':
        files = request.files['file'] # something like this (check link onn discord)
        # print(files)
        # with open(os.path.abspath(f'./test-pipi-caca/{files}'), 'wb') as f:
        #     f.write(files.content)
        # save file with os module
        app.logger.info('HERE IS WHAT FILES IS:')
        app.logger.info(files)
        app.logger.info('HERE IS THE TYPE')
        app.logger.info(type(files))
        emotionResult = processAudio(files)

        response = jsonify(emotionResult)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        ## main

if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=3001)
    emotion = processAudio("./assets/hello.wav")
    print(emotion)
