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