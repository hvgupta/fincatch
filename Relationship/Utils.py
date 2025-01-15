from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from urllib.parse import unquote

def removeUneededChars(string: str) -> str:
    return ''.join(c for c in string if c.isalnum() or c.isspace() or c == '-')

def removeStopWordsandStem(string: str) -> str:
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = string.split()
    return ' '.join(ps.stem(removeUneededChars(w)) for w in words if w not in stop_words)

def getNameFromUrl(url: str) -> str:
    name = url.split("/")[-1]
    if "investopedia" in url:
        name = name[:-4]
    return unquote(name)
