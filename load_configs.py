import sys
sys.path.append("llmapslab")
import os
import json

with open('../secrets.json', 'r') as jsonfile:
    configs = json.load(jsonfile)
openai_api_key=configs.get('openai_api_key')
llama_api_key = configs.get('llama_api_key')
ai21_api_key = configs.get('ai21_api_key')
gemini_api_key = configs.get('gemini_api_key')