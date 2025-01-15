from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from urllib.parse import unquote

def removeStopWordsandStem(string:str)->str:
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = string.split()
    return ' '.join([ps.stem(w) for w in words if w not in stop_words])

def getNameFromUrl(url:str)->str:
    return unquote(url.split("/")[-1])
