from bs4 import BeautifulSoup
from utility import fetch

def __getSoupObject(parsedString) -> BeautifulSoup:
    return BeautifulSoup(parsedString, 'html.parser')

def __getText(page:BeautifulSoup) -> str:
    text = page.find_all("div", class_="comp mntl-sc-page mntl-block article-body-content")
    text = ' '.join([t.get_text() for t in text])
    
    return text

def __getSummary(page:BeautifulSoup) -> str:
    summary = page.find('meta', {'name': 'description'})['content'] if page.find('meta', {'name': 'description'}) else ''
    
    return summary


async def getTextandSummary(url:str) -> tuple[str, str]:
    reponse = await fetch(url)
    page = __getSoupObject(reponse)
    text = __getText(page)
    summary = __getSummary(page)
    
    return text, summary