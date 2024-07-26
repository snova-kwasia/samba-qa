import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))
sys.path.append(kit_dir)
sys.path.append(repo_dir)
from dotenv import load_dotenv
import re

# Join the repo_dir with the .env file path
env_path = os.path.join(repo_dir, '.env')

# Load the .env file
load_dotenv(env_path, override=True)

import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(token= os.getenv("SLACK_TOKEN", ""))

# SambaQA API configuration
SAMBAQA_API_URL = os.getenv("SAMBAQA_API_URL", "http://localhost:8000/retrievers/basic-rag/answer")
DEFAULT_COLLECTION = "snspubdocumentation"
DEFAULT_MODEL = "sambastudio/Meta-Llama-3-70B-Instruct"

def query_sambaqa(question):
    payload = {
        "collection_name": DEFAULT_COLLECTION,
        "query": question,
        "model_configuration": {
            "name": DEFAULT_MODEL,
            "parameters": {
                "temperature": 0.1,
                "do_sample": False,
                "max_tokens_to_generate": 1024,
                "select_expert": DEFAULT_MODEL.split('/')[-1],
                "process_prompt": False
            }
        },
        "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|> \nYou are an AI assistant specialising in information retrieval and analysis. You are thorough and give detailed step by step answers to question and you must ALWAYS quote sources where you find them. Answer the following question based only on the given context: \nContext: {context} \nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
        "retriever_name": "vectorstore",
        "retriever_config": {
            "search_type": "similarity",
            "search_kwargs": {
                "k": 10
            },
            "filter": {}
        },
        "stream": False
    }

    try:
        response = requests.post(SAMBAQA_API_URL, json=payload)
        response.raise_for_status()
        return response.json()['answer']
    except requests.RequestException as e:
        return f"Error querying SambaQA: {str(e)}"

@app.event("message")
def handle_message(message, say, logger):
    try:
        if message['channel_type'] == 'im':
            question = message['text']
            cleaned_question = re.sub(r'<@U[A-Z0-9]+>\s*', '', question)
            response = query_sambaqa(cleaned_question)
            say(response)
    except Exception as e:
        error_message = "SambaQA is experiencing difficulties, hang tight while we investigate."
        logger.error(f"Error in handle_message: {str(e)}")
        say(error_message)

@app.event("app_mention")
def handle_app_mention_events(body, logger, say):
    try:
        question = body["event"]["text"]
        cleaned_question = re.sub(r'<@U[A-Z0-9]+>\s*', '', question)
        response = query_sambaqa(cleaned_question)
        say(response)
        logger.info(body)
    except Exception as
        error_message = "SambaQA is experiencing difficulties, hang tight while we investigate."
        logger.error(f"Error in handle_app_mention_events: {str(e)}")
        say(error_message)
        
# Start your app
if __name__ == "__main__":
    SocketModeHandler(app,  os.getenv("SLACK_APP_TOKEN", "")).start()

    