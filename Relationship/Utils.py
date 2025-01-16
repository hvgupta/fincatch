from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from urllib.parse import unquote

def removeUneededChars(string: str) -> str:
    """
    Remove invalid characters from a string.

    This function filters a string, retaining only alphanumeric characters, 
    spaces, and hyphens. All other characters are removed.

    Parameters:
    string (str): The input string from which to remove invalid characters.

    Returns:
    str: A string containing only alphanumeric characters, spaces, and hyphens.
    """
    return ''.join(c for c in string if c.isalnum() or c.isspace() or c == '-')

def removeStopWordsandStem(string: str) -> str:
    """
    Remove stop words from a string and apply stemming to the remaining words.

    This function takes a string, splits it into words, removes common English 
    stop words, and stems the remaining words using the Porter stemming algorithm.

    Parameters:
    string (str): The input string from which to remove stop words and stem words.

    Returns:
    str: A string containing the processed words, with stop words removed and 
         remaining words stemmed.
    """
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = string.split()
    return ' '.join(ps.stem(removeUneededChars(w)) for w in words if w not in stop_words)

def getNameFromUrl(url: str) -> str:
    """
    Extract the name component from a URL.

    This function takes a URL string, splits it to retrieve the last segment, 
    and adjusts the name if the URL is from Investopedia by removing the 
    last four characters (typically the file extension). It then decodes 
    any URL-encoded characters.

    Parameters:
    url (str): The URL from which to extract the name.

    Returns:
    str: The decoded name extracted from the URL.
    """
    name = url.split("/")[-1]
    if "investopedia" in url:
        name = name[:-4]
    return unquote(name)
