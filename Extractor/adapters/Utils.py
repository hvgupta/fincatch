import aiohttp


async def fetch(url) -> str:
     """
    Asynchronously fetch the raw content of a webpage.

    This function retrieves the content of a webpage from the specified URL 
    using an asynchronous HTTP GET request. It reads the response in chunks 
    to efficiently handle large amounts of data, ensuring that the entire 
    content is fetched and returned as a single string.

    Parameters:
    url (str): The URL of the webpage to fetch.

    Returns:
    str: The raw HTML content of the webpage as a string.
    """
    async with aiohttp.ClientSession() as session: 
        async with session.get(url) as response:
            totalResponse = ""
            while True:
                chunk = await response.content.read(1024)  # the text is read in chunks to consider the cases where the amount of data is very large
                if not chunk:
                    break
                totalResponse += chunk.decode('utf-8', errors='ignore')
    
    return totalResponse


def writeToFile_GatheredContent(store, filename):
     """
    Write gathered content and summaries to a specified text file.

    This function takes a dictionary containing URLs as keys and their 
    corresponding content and summaries as values, then writes this 
    information to a text file. The format includes the URL, summary, 
    and text, separated by headers for clarity.

    Parameters:
    store (dict): A dictionary where each key is a URL and the value is 
                  an object containing 'summary' and 'text' attributes.
    filename (str): The path to the file where the content will be stored.
    """
    with open(filename, 'w') as f:
        for key in store:
            f.write(f"URL: {key}\n")
            f.write("--------- Summary ---------\n")
            f.write(store[key].summary + "\n")
            f.write("--------- Text ---------\n")
            f.write(store[key].text + "\n")
            f.write('============================================\n')
