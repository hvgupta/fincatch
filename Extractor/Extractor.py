import pandas as pd
import asyncio
import aiohttp
from dataclasses import dataclass
import adapters.investopedia_adapter as investopedia
import adapters.wikipedia_adapter as enWiki


@dataclass(frozen=True)
class Info:
    url: str
    text: str
    summary: str 

store = {}

async def fetch(url) -> dict[str,Info]:
    text = ""
    summary = ""
    
    if 'wikipedia' in url:
        text, summary = await enWiki.getTextandSummary(url)
    elif 'investopedia' in url:
        text, summary = await investopedia.getTextandSummary(url)

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
    with open('output.txt', 'w') as f:
        for key in store:
            f.write(f"URL: {key}\n")
            f.write("--------- Summary ---------\n")
            f.write(store[key].summary + "\n")
            f.write("--------- Text ---------\n")
            f.write(store[key].text + "\n")
            f.write('============================================\n')

if __name__ == "__main__":
    asyncio.run(main())
