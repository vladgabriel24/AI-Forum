# In this script, I will make a quick PoC for the workflow proposed


import requests
import configparser

import spacy

hf_config = configparser.ConfigParser()
hf_config.read('hf.ini')

hf_token = hf_config['HuggingFace']['token']

headers = {
    "Authorization": f"Bearer {hf_token}",
}

def query(model_API, payload):
    response = requests.post(model_API, headers=headers, json=payload)
    return response.json()

API_URL_Cerebras = "https://router.huggingface.co/cerebras/v1/chat/completions"
API_URL_TogetherAI = "https://router.huggingface.co/together/v1/chat/completions"

prompt = "When was the last president election in Spain?"

response_LLaMA = query(API_URL_Cerebras, {
    "messages": [
        {
            "role": "user",
            "content": f"{prompt}"
        }
    ],
    "model": "llama-3.3-70b"
})["choices"][0]["message"]["content"]

response_DeepSeek = query(API_URL_TogetherAI, {
    "messages": [
        {
            "role": "user",
            "content": f"{prompt}"
        }
    ],
    "model": "deepseek-ai/DeepSeek-R1"
})["choices"][0]["message"]["content"]

response_Mistral = query(API_URL_TogetherAI, {
    "messages": [
        {
            "role": "user",
            "content": f"{prompt}"
        }
    ],
    "model": "mistralai/Mistral-7B-Instruct-v0.3"
})["choices"][0]["message"]["content"]


print(response_LLaMA, response_DeepSeek, response_Mistral)

# Load a pretrained model
nlp = spacy.load("en_core_web_sm")

# Run aspect extraction
aspects_LaMA = [chunk.text for chunk in nlp(response_LLaMA).noun_chunks]
aspects_DeepSeek = [chunk.text for chunk in nlp(response_DeepSeek).noun_chunks]
aspects_Mistral = [chunk.text for chunk in nlp(response_Mistral).noun_chunks]


# common_aspects = set(aspects_LaMA).intersection(set(aspects_DeepSeek).intersection(set(aspects_Mistral))) TO DO
