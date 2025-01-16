import aiohttp


async def fetch(url) -> str:
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
    with open(filename, 'w') as f:
        for key in store:
            f.write(f"URL: {key}\n")
            f.write("--------- Summary ---------\n")
            f.write(store[key].summary + "\n")
            f.write("--------- Text ---------\n")
            f.write(store[key].text + "\n")
            f.write('============================================\n')