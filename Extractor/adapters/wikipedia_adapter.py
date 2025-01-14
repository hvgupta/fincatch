from bs4 import BeautifulSoup
from utility import fetch

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

def __getSummary(page: BeautifulSoup) -> str:
    content = __getContent(page)
    if content == None:
        return ""
    summary_soup = content.find_all('p')
    summary = summary_soup[1].get_text() if len(summary_soup)>1 else ''
    
    return summary

async def getTextandSummary(url: str) -> tuple[str, str]:
    reponse = await fetch(url)
    page = __getSoupObject(reponse)
    text = __getText(page)
    summary = __getSummary(page)
    
    return text, summary