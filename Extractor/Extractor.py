import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import asyncio
import adapters.Investopedia_Adapter as investopedia
import adapters.Wikipedia_Adapter as enWiki
import json

async def __fetch(url) -> list:
    """
    Asynchronously fetch content and summary from a specified URL.

    This function retrieves the content and summary of a webpage based on 
    the provided URL. It supports fetching data from Wikipedia and 
    Investopedia. Depending on the URL, it delegates the fetching of 
    content to the appropriate asynchronous method.

    Parameters:
    url (str): The URL of the webpage from which to fetch content. 
               It should be a Wikipedia or Investopedia link.

    Returns:
    list: A list containing the following elements:
        - url (str): The original URL.
        - text (str): The full text content of the webpage.
        - summary (str): A brief summary of the content.
   
    """
    text = ""
    summary = ""
    
    if 'wikipedia' in url:
        text, summary = await enWiki.getPageContent(url)
    elif 'investopedia' in url:
        text, summary = await investopedia.getPageContent(url)

    return [url, text, summary]
    
async def __url_Iterator(filename:str)->dict:
    """
    Asynchronously iterate over URLs in a CSV file and fetch their content and summaries.

    This function reads a CSV file containing URLs, retrieves the content and 
    summaries of each URL using the `__fetch` function, and compiles the results 
    into a dictionary. Each URL is associated with its corresponding text content 
    and summary.

    Parameters:
    filename (str): The path to the CSV file containing a column labeled 'URL' 
                    with the URLs to be processed.

    Returns:
    dict: A dictionary where each key is a URL and the value is another dictionary 
          containing the following:
        - 'text' (str): The full text content of the webpage.
        - 'summary' (str): A brief summary of the content.
    """
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
            
def getURLContent(csv_file_path:str, output_file_path:str)->dict:
    """
    Synchronously fetch content and summaries for URLs listed in a CSV file.

    This function serves as a wrapper to create an event loop and execute
    the asynchronous `__url_Iterator` function, which retrieves the content 
    and summaries for each URL found in the specified CSV file.

    Parameters:
    filename (str): The path to the CSV file containing a column labeled 'URL' 
                    with the URLs to be processed.

    Returns:
    dict: A dictionary where each key is a URL and the value is another dictionary 
          containing the following:
        - 'text' (str): The full text content of the webpage.
        - 'summary' (str): A brief summary of the content.
    """
    loop = asyncio.get_event_loop()
    output = loop.run_until_complete(__url_Iterator(csv_file_path))
    with open(output_file_path, 'w') as f:
        json.dump(output, f)
