import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import pickle
from sklearn import model_selection
warnings.filterwarnings('ignore')


paths = []
labels = []
for dirname, _, filenames in os.walk('./data'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
    if len(paths) == 2800:
        break

print('Dataset is Loaded')

len(paths)

paths[:5]

labels[:5]

## Create a dataframe
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()

df['label'].value_counts()

sns.countplot(df['label'])

def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()
    
def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()

showPlots = input("Show plots?: (y/n) ")
if (showPlots == "y"):
    inp = input("Fear?: (y/n) ")
    if (inp == "y"):
        emotion = 'fear'
        path = np.array(df['speech'][df['label']==emotion])[0]
        data, sampling_rate = librosa.load(path)
        waveplot(data, sampling_rate, emotion)
        spectogram(data, sampling_rate, emotion)
        Audio(path, autoplay=True)

    inp = input("Angry?: (y/n)")
    if (inp == "y"):
        emotion = 'angry'
        path = np.array(df['speech'][df['label']==emotion])[1]
        data, sampling_rate = librosa.load(path)
        waveplot(data, sampling_rate, emotion)
        spectogram(data, sampling_rate, emotion)
        Audio(path)

    inp = input("Disgust?: (y/n)")
    if (inp == "y"):
        emotion = 'disgust'
        path = np.array(df['speech'][df['label']==emotion])[0]
        data, sampling_rate = librosa.load(path)
        waveplot(data, sampling_rate, emotion)
        spectogram(data, sampling_rate, emotion)
        Audio(path)

    inp = input("Neutral?: (y/n)")
    if (inp == "y"):
        emotion = 'neutral'
        path = np.array(df['speech'][df['label']==emotion])[0]
        data, sampling_rate = librosa.load(path)
        waveplot(data, sampling_rate, emotion)
        spectogram(data, sampling_rate, emotion)
        Audio(path)

    inp = input("Sad?: (y/n)")
    if (inp == "y"):
        emotion = 'sad'
        path = np.array(df['speech'][df['label']==emotion])[0]
        data, sampling_rate = librosa.load(path)
        waveplot(data, sampling_rate, emotion)
        spectogram(data, sampling_rate, emotion)
        Audio(path)

    inp = input("Ps?: (y/n)")
    if (inp == "y"):
        emotion = 'ps'
        path = np.array(df['speech'][df['label']==emotion])[0]
        data, sampling_rate = librosa.load(path)
        waveplot(data, sampling_rate, emotion)
        spectogram(data, sampling_rate, emotion)
        Audio(path)

    inp = input("Happy?: (y/n)")
    if (inp == "y"):
        emotion = 'happy'
        path = np.array(df['speech'][df['label']==emotion])[0]
        data, sampling_rate = librosa.load(path)
        waveplot(data, sampling_rate, emotion)
        spectogram(data, sampling_rate, emotion)
        Audio(path)


def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc
    
    

print(extract_mfcc(df['speech'][0]))

X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))


X = [x for x in X_mfcc]
X = np.array(X)
print(X.shape)

## input split
X = np.expand_dims(X, -1)
print(X.shape)
# print(X)


# AI Model here


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])
print(df[['label']])

# print(y)
y = y.toarray()

print(y)
# print(y.shape)

model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40,1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.10)

# Train the model
history = model.fit(X_train, Y_train, validation_split=0.2, epochs=50, batch_size=64)
fileName = "speechModel.pkl"

# export and import model for furture use
pickle.dump(model, open(fileName, 'wb'))
loadModel = pickle.load(open(fileName, 'rb'))
# test model
result = loadModel.evaluate(X_test, Y_test)
print("X_Test: \n", X_test)
print("Y_Test: \n", Y_test)
print("lengths (x,y): ", len(X_test), len(Y_test))
print("len of xtest:", len(X_test[0]))


print(loadModel.predict([X_test[0]]))


# epochs = list(range(50))
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# plt.plot(epochs, acc, label='train accuracy')
# plt.plot(epochs, val_acc, label='val accuracy')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend()
# plt.show()

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.plot(epochs, loss, label='train loss')
# plt.plot(epochs, val_loss, label='val loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()
# plt.show()