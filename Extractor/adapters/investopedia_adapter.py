import sys
import os
sys.path.append(os.path.dirname(__file__))

from bs4 import BeautifulSoup
from utility import fetch
from ai_adapter import generateSummary

def __getSoupObject(parsedString) -> BeautifulSoup:
    return BeautifulSoup(parsedString, 'html.parser')

def __getText(page:BeautifulSoup) -> str:
    text = page.find_all("div", class_="comp mntl-sc-page mntl-block article-body-content")
    text = ' '.join([t.get_text() for t in text])
    
    return text
async def getPageContent(url:str) -> tuple[str, str]:
    reponse = await fetch(url)
    page = __getSoupObject(reponse)
    text = __getText(page)
    summary = generateSummary(text)
    
    return text, summary