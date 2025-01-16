import sys
import os
sys.path.append(os.path.dirname(__file__))

from bs4 import BeautifulSoup
from Utils import fetch
from AI_Adapter import generateSummary

def __getSoupObject(parsedString: str) -> BeautifulSoup:
     """
    Create a BeautifulSoup object from a parsed HTML string.

    This function takes a string containing parsed HTML content and converts it 
    into a BeautifulSoup object, which allows for easy manipulation and 
    navigation of the HTML structure.

    Parameters:
    parsedString (str): A string representation of parsed HTML content.

    Returns:
    BeautifulSoup: A BeautifulSoup object that represents the HTML structure 
                   for further processing.
    """
    return BeautifulSoup(parsedString, 'html.parser')

def __getContent(page: BeautifulSoup) -> BeautifulSoup:
     """
    Extracts the content <div> from a BeautifulSoup page object.

    This function searches for a <div> element with a specific id and class 
    within the provided BeautifulSoup page object. It returns the found 
    <div> element if it exists; otherwise, it returns None.

    Parameters:
    page (BeautifulSoup): A BeautifulSoup object representing the parsed HTML page.

    Returns:
    BeautifulSoup or None: The <div> element containing the content if found, 
                           or None if the element does not exist.
    """
    content_div = page.find('div', id='mw-content-text', class_='mw-body-content')
    return content_div if content_div else None

def __getText(page: BeautifulSoup) -> str:
    """
    Extract the main text content from a BeautifulSoup object.

    This function takes a BeautifulSoup object representing a parsed HTML 
    page and retrieves the text from all paragraph (`<p>`) elements. It 
    calls the `__getContent` function to obtain the relevant content 
    section before extracting the text, ensuring that only the main 
    body text is returned.

    Parameters:
    page (BeautifulSoup): A BeautifulSoup object representing the parsed 
                          HTML of the webpage.

    Returns:
    str: The extracted text content from the paragraph elements. If no 
         content is found, an empty string is returned.
    """
    content = __getContent(page)
    if content == None:
        return ""
    text = content.find_all('p')
    text = ' '.join([t.get_text() for t in text]) if text else ''
    
    return text

async def getPageContent(url: str) -> tuple[str, str]:
    """
    Asynchronously fetch and extract content and summary from a webpage.

    This function retrieves the HTML content from the specified URL, processes 
    it to extract the main text, and generates a summary of that text. It 
    utilizes the `fetch` function to get the page content, then uses helper 
    functions to parse the HTML and summarize the text.

    Parameters:
    url (str): The URL of the webpage from which to fetch content.

    Returns:
    tuple[str, str]: A tuple containing:
        - text (str): The full text content extracted from the webpage.
        - summary (str): A concise summary of the extracted text.
    """
    reponse = await fetch(url)
    page = __getSoupObject(reponse)
    text = __getText(page)
    summary = generateSummary(text)
    
    return text, summary
