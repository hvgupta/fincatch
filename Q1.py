import pandas as pd
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import dotenv
from dataclasses import dataclass


@dataclass(frozen=True)
class Info:
    url: str
    text: str
    summary: str 

dotenv.load_dotenv(dotenv.find_dotenv())

store = {}

async def addString(url, string):
    if url in store:
        info = store[url]
        store[url] = Info(url, info.text + string, info.summary)
        

    
def convertSoup(parsedString) -> BeautifulSoup:
    return BeautifulSoup(parsedString, 'html.parser')

async def fetch(url) -> None:
    text = ""
    summary = ""
    if 'wikipedia' in url:
        page = enWiki.page(url.split('/')[-1])
        text = page.text
        summary = page.summary
    else:
        async with aiohttp.ClientSession() as session: 
            async with session.get(url) as response:
                totalResponse = ""
                while True:
                    chunk = await response.content.read(1024)  # the text is read in chunks to consider the cases where the amount of data is very large
                    if not chunk:
                        break
                    totalResponse += chunk.decode('utf-8', errors='ignore')
                    
        page = convertSoup(totalResponse)     
        text = page.find_all("div", class_="comp mntl-sc-page mntl-block article-body-content")
        text = ' '.join([t.get_text() for t in text])
        summary = page.find('meta', {'name': 'description'})['content'] if page.find('meta', {'name': 'description'}) else ''
    
    store[url] = Info(url, text, summary)
    
async def outerFunc():
    data = pd.read_csv('FinCatch_Sources_Medium.csv')
    for _, row in data.iterrows():
        url = row['URL']
        store[url] = Info(url, "", "")
        asyncio.create_task(fetch(url)) 
            
def main():
    outerFunc()
    with open('output.txt', 'w') as f:
        for key in store:
            f.write(f"URL: {key}\n")
            f.write("--------- Summary ---------\n")
            f.write(store[key].summary + "\n")
            f.write("--------- Text ---------\n")
            f.write(store[key].text + "\n")
            f.write('============================================\n')

if __name__ == "__main__":
    main()
