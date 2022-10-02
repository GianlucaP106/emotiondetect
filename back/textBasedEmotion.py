import pandas as pd
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import requests
import nltk
from LeXmo import LeXmo
nltk.download('omw-1.4')
nltk.download('punkt')

inputTextModel = """I am glad to see you."""
emo = LeXmo.LeXmo(inputTextModel)
print(emo)
