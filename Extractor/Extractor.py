import pandas as pd
import asyncio
from dataclasses import dataclass
import adapters.investopedia_adapter as investopedia
import adapters.wikipedia_adapter as enWiki
from utility import writeToFile_GatheredContent


@dataclass
class Info:
    url: str
    text: str
    summary: str 

store = {}

async def fetch(url) -> dict[str,Info]:
    text = ""
    summary = ""
    
    if 'wikipedia' in url:
        text, summary = await enWiki.getPageContent(url)
    elif 'investopedia' in url:
        text, summary = await investopedia.getPageContent(url)

    store[url] = Info(url, text, summary)
    
async def outerFunc():
    data = pd.read_csv('FinCatch_Sources_Medium.csv')
    tasks = []
    for _, row in data.iterrows():
        url = row['URL']
        store[url] = Info(url, "", "")
        tasks.append(fetch(url))
    await asyncio.gather(*tasks)
            
async def main():
    await outerFunc()
    writeToFile_GatheredContent(store, 'output.txt')

if __name__ == "__main__":
    asyncio.run(main())
