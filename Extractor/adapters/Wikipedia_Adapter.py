import sys
import os
sys.path.append(os.path.dirname(__file__))

from bs4 import BeautifulSoup
from Utils import fetch
from AI_Adapter import generateSummary

def __getSoupObject(parsedString: str) -> BeautifulSoup:
    return BeautifulSoup(parsedString, 'html.parser')

def __getContent(page: BeautifulSoup) -> BeautifulSoup:
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
