import sys
import os
sys.path.append(os.path.dirname(__file__))

from bs4 import BeautifulSoup
from Utils import fetch
from AI_Adapter import generateSummary

def __getSoupObject(parsedString) -> BeautifulSoup:
    return BeautifulSoup(parsedString, 'html.parser')

def __getText(page:BeautifulSoup) -> str:
    text = page.find_all("div", class_="comp mntl-sc-page mntl-block article-body-content")
    text = ' '.join([t.get_text() for t in text])
    
    return text
async def getPageContent(url:str) -> tuple[str, str]:
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
