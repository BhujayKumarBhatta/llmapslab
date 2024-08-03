from  openai import OpenAI
from load_configs import openai_api_key
client = OpenAI(api_key=openai_api_key,
                # organization='Personal',
                # project='proj_DqdKUU38qOGVj9Qxdq6UlXcD',
               )

def call_openai_api(messages, client=client, model="gpt-4o-mini", max_retries=5, wait_time=5):
    retries = 0
    while retries < max_retries:
        try:
            # Make a request to the OpenAI ChatCompletion API
           response = client.chat.completions.create(
              model=model,
              messages=messages,
           )
           return response
        except client.error.RateLimitError:
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    raise Exception("Max retries exceeded. Please check your plan and billing details.")
