import wikipediaapi
import os
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

client = OpenAI(
    base_url="https://hkust.azure-api.net",
    api_key=os.getenv("HKUST_AZURE_AI_API_KEY_PRIMARY"),
    organization="hkust",
)


enWiki = wikipediaapi.Wikipedia('test/1.0 (hvgupta@outlook.in)','en' )

page = enWiki.page("Currency")
with open('output.txt', 'w') as f:
    f.write(page.text)


with open("summary.txt", "w") as f:
    f.write(page.summary)
    
response = client.Completion.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the currency?"
        }
    ],
    model="text-davinci-003",
)
print(response)