import os
from openai import AzureOpenAI
import dotenv

dotenv.load_dotenv()

client = AzureOpenAI(
    azure_endpoint="https://hkust.azure-api.net",
    api_key=os.getenv("HKUST_AZURE_AI_API_KEY_PRIMARY"),
    api_version="2023-05-15"
)

def generateSummary(parsedString:str)->str:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "assistant",
                "content": "You are a financial expert."
            },
            {
                "role": "user",
                "content": f"generate a summary for this financial article:\n{parsedString}"
            }
        ],
        model="gpt-4o",
    )
    return response.choices[0].message.content