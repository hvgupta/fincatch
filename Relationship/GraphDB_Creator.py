import json
from Relationship.Utils import *
from Relationship.Database import Data_Access_Facade

def load_output(file:str)->dict:
    with open(file, "r") as outputFile:
        output = json.load(outputFile)
    return output

def getWebsiteInfo(inputFile:str):
    output = load_output(inputFile)
    for key in output:
        yield key, removeStopWordsandStem(output[key])

def createGraphDB(inputFile:str):
    for key, value in getWebsiteInfo(inputFile):
        status = Data_Access_Facade.create_node("Website",{"name":getNameFromUrl(key), "url":key})
        for text_key, text_val in value.items():
            pass