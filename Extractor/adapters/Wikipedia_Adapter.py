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
    content = __getContent(page)
    if content == None:
        return ""
    text = content.find_all('p')
    text = ' '.join([t.get_text() for t in text]) if text else ''
    
    return text

async def getPageContent(url: str) -> tuple[str, str]:
    reponse = await fetch(url)
    page = __getSoupObject(reponse)
    text = __getText(page)
    summary = generateSummary(text)
    
    return text, summary