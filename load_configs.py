import sys
sys.path.append("llmapslab")
import os
import json

# with open('../secrets.json', 'r') as jsonfile:
#     configs = json.load(jsonfile)
# openai_api_key=configs.get('openai_api_key')
# llama_api_key = configs.get('llama_api_key')
# ai21_api_key = configs.get('ai21_api_key')
# gemini_api_key = configs.get('gemini_api_key')

from pydantic import BaseModel, SecretStr

class ApiConfig(BaseModel):
    openai_api_key: SecretStr
    llama_api_key: SecretStr
    ai21_api_key: SecretStr
    gemini_api_key: SecretStr
    tavily_api_key: SecretStr

    def __str__(self):
        return "ApiConfig(secrets hidden)"

import json

with open('../secrets.json', 'r') as jsonfile:
    configs = json.load(jsonfile)

api_config = ApiConfig(
    openai_api_key=configs.get('openai_api_key'),
    llama_api_key=configs.get('llama_api_key'),
    ai21_api_key=configs.get('ai21_api_key'),
    gemini_api_key=configs.get('gemini_api_key'),
    tavily_api_key=configs.get('tavily_api_key')
)

openai_api_key=api_config.openai_api_key
llama_api_key=api_config.llama_api_key
ai21_api_key=api_config.ai21_api_key
gemini_api_key=api_config.gemini_api_key
tavily_api_key=api_config.tavily_api_key
