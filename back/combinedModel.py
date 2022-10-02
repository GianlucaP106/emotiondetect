import os
from sklearn import model_selection
import speech_recognition as sr
import numpy as np
import librosa
import pickle
import xgboost as xgb
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import nltk
from LeXmo import LeXmo
nltk.download('omw-1.4')
nltk.download('punkt')
# from index import convertToText
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
fileName = "./pkl/speechModel.pkl"
loadModel = pickle.load(open(fileName, 'rb'))

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

### Converting speech to text format

## process data
# LUT = { "fear": 0, "happy": 1, "neutral": 2, "sad": 3, "angry": 4, "disgust": 5, "surprise": 6 }
X = []
y = []
pathToData = "./assets/combined_model_data"
emotionLUT = { "FEA": 0, "HAP": 1, "NEU": 2, "SAD": 3, "ANG": 4, "DIS": 5 }
index = 0
for audio in os.listdir(pathToData):
    # extract data
    try:
        index += 1
        print(index)
        if (index == 1900): break
        inpForSpeech = extract_mfcc(pathToData + "/" + audio)
        formattedInput = convertToFormat(inpForSpeech)
        prediction = loadModel.predict(formattedInput)
        inputTextModel = convertToText(pathToData + "/" + audio) 
        emo = LeXmo.LeXmo(inputTextModel) 
        # standardize data
        standardized = prepCombinedModel(prediction, emo)
        # retrieve answer (1001_DFA_ANG_XX)
        answer = emotionLUT[audio[-10:-7]]
        y.append(answer)
        # format X
        x = [i for i in standardized.values()]
        X.append(x)
    except:
        print("skipping data point")
x_file = open("./combinedModelData/x-new.txt", 'w')
x_file.write(str(X))
y_file = open("./combinedModelData/y-new.txt", 'w')
y = np.array(y)
y_file.write(str(y))

# #t-fear
# #t-happy
# #t-neutral
# #t-sad
# #t-angry
# #t-disgust
# #t-surprise

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.15)

train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)

params = {
    # 'max-_depth': 4,
    'eta': 0.3,
    'objective': 'multi:softmax',
    'num_class': 6
}
epochs = 20

model = xgb.train(params, train, epochs)

testdata = [[9.9999082e-01, 1.1481602e-06, 7.6124115e-07, 2.3836126e-06, 4.3731538e-09, 4.6895557e-06, 1.4864995e-07, 1.2145509e-06, 9.9998903e-01, 4.2692032e-09]]
testdata = np.array(testdata)
test1 = xgb.DMatrix(testdata)


# export and import model for furture use
fileName = "./pkl/combinedModel.pkl"
pickle.dump(model, open(fileName, 'wb'))
loadModel = pickle.load(open(fileName, 'rb'))
print(loadModel.predict(test1))