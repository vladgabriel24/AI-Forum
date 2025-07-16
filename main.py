# In this script, I will make a quick PoC for the workflow proposed


import requests
import configparser

hf_config = configparser.ConfigParser()
hf_config.read('hf.ini')

hf_token = hf_config['HuggingFace']['token']

API_URL = "https://router.huggingface.co/cerebras/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {hf_token}",
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

response = query({
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    "model": "llama-3.3-70b"
})

print(response["choices"][0]["message"])