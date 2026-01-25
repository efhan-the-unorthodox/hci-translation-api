import requests
import os
import dotenv

dotenv.load_dotenv()

API_URL = "https://router.huggingface.co/v1/chat/completions"
api_key = os.getenv("HF_API_KEY")
if not api_key:
    raise RuntimeError("HF_API_KEY not set in environment or .env file")

headers = {"Authorization": f"Bearer {api_key}"}

payload = {
    "messages": [
        {
            "role": "system",
            "content": "You are a Professional Translator. Your job is to translate given sentences from English to Chinese (Simplified). You should only return the translation in the response.",
        },
        {
            "role": "user",
            "content": "Translate: Good interpersonal relationships make people healthier and happier.",
        },
    ],
    "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B:featherless-ai",
}

response = requests.post(API_URL, headers=headers, json=payload)
print(response.json()['choices'][0]['message'])
