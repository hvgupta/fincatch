import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from Utils import *
from Database import Data_Access_Facade
from time import sleep

def __load_output(file:str)->dict:
    with open(file, "r") as outputFile:
        output = json.load(outputFile)
    return output

def __getWebsiteInfo(inputFile:str):
    output = __load_output(inputFile)
    for key in output:
        yield key, {"text":removeStopWordsandStem(output[key]["text"]), "summary":removeStopWordsandStem(output[key]["summary"])}

def createGraphDB(inputFile:str):
    for key, value in __getWebsiteInfo(inputFile):
        Data_Access_Facade.createNode("Website",{"name":getNameFromUrl(key), "url":key})
        for text_key, text_val in value.items():
            stemmedText = removeStopWordsandStem(text_val)
            set_semmedText = set(stemmedText.split())
            for word in set_semmedText:
                Data_Access_Facade.createNode("Word",{"name":word})
                Data_Access_Facade.createRelationship("Word","Website",{"name":word},{"name":getNameFromUrl(key)},text_key,{"count":text_val.count(word)})
                sleep(0.1)