import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import asyncio
import Extractor.adapters.Investopedia_Adapter as investopedia
import Extractor.adapters.Wikipedia_Adapter as enWiki

async def __fetch(url) -> list:
    text = ""
    summary = ""
    
    if 'wikipedia' in url:
        text, summary = await enWiki.getPageContent(url)
    elif 'investopedia' in url:
        text, summary = await investopedia.getPageContent(url)

    return [url, text, summary]
    
async def __url_Iterator(filename:str)->dict:
    data = pd.read_csv(filename)
    tasks = []
    jsonOutput = {}
    for _, row in data.iterrows():
        url = row['URL']
        tasks.append(__fetch(url))
    results = await asyncio.gather(*tasks)
    
    for result in results:
        url, text, summary = result
        jsonOutput[url] = {'text': text, 'summary': summary}
    
    return jsonOutput
            
def getURLContent(filename:str)->dict:
    loop = asyncio.get_event_loop()
    output = loop.run_until_complete(__url_Iterator(filename))
    return output
