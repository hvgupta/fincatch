import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from Utils import *
from Database import Data_Access_Facade
from time import sleep

def load_output(file:str)->dict:
     """
    Load JSON data from a specified file.

    This function reads a JSON file and parses its contents into a Python 
    dictionary. It expects the file to be in a valid JSON format.

    Parameters:
    file (str): The path to the JSON file to be loaded.

    Returns:
    dict: A dictionary containing the parsed JSON data.
    """
    with open(file, "r") as outputFile:
        output = json.load(outputFile)
    return output

def getWebsiteInfo(inputFile:str):
     """
    Generate website information from a JSON input file.

    This function reads a JSON file containing website data, processes the 
    text and summary of each entry by removing stop words and stemming, 
    and yields the processed information.

    Parameters:
    inputFile (str): The path to the JSON input file.

    Yields:
    tuple: A tuple containing the key and a dictionary with processed 
           "text" and "summary" for each entry in the JSON.
    """
    output = load_output(inputFile)
    for key in output:
        yield key, {"text":removeStopWordsandStem(output[key]["text"]), "summary":removeStopWordsandStem(output[key]["summary"])}

def createGraphDB(inputFile:str):
    """
    Create a graph database from website information provided in a JSON input file.

    This function processes each entry in the input JSON file, creates nodes 
    for websites and unique words, and establishes relationships between them. 
    It uses the `getWebsiteInfo` function to retrieve the website data and 
    creates nodes and relationships using the `Data_Access_Facade`.

    Parameters:
    inputFile (str): The path to the JSON input file containing website data.

    Returns:
    None: This function does not return a value. It performs actions on the 
          graph database.
    """
    for key, value in getWebsiteInfo(inputFile):
        Data_Access_Facade.createNode("Website",{"name":getNameFromUrl(key), "url":key})
        for text_key, text_val in value.items():
            stemmedText = removeStopWordsandStem(text_val)
            set_semmedText = set(stemmedText.split())
            for word in set_semmedText:
                Data_Access_Facade.createNode("Word",{"name":word})
                Data_Access_Facade.createRelationship("Word","Website",{"name":word},{"name":getNameFromUrl(key)},text_key,{"count":text_val.count(word)})
                sleep(0.1)
