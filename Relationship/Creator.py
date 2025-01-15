import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download("stopwords")

def removeStopWordsandStem(string:str)->str:
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = string.split()
    return ' '.join([ps.stem(w) for w in words if w not in stop_words])


def load_output(file:str)->dict:
    outputFile = open(file, "r")
    output = json.load(outputFile)
    
    return output

def createDatabase(inputFile:str)->None:
    with open("test_output.txt", "w") as f:
        for key, value in load_output(inputFile).items():
            stemmedText = removeStopWordsandStem(value["text"])
            stemmedSummary = removeStopWordsandStem(value["summary"])
            f.write(f"{key}: {stemmedText}\n{stemmedSummary}\n\n")