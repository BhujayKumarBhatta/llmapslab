import sys
sys.path.append("llmapslab")
import os
import json

with open('../secrets.json', 'r') as jsonfile:
    configs = json.load(jsonfile)
openai_api_key=configs.get('openai_api_key')
