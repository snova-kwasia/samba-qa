from dotenv import load_dotenv
import os
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from backend.settings import settings

app = App(token=settings.SLACK_TOKEN)

# SambaQA API configuration
SAMBAQA_API_URL = "http://localhost:8000/retrievers/basic-rag/answer"
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
def handle_message(message, say):
    if message['channel_type'] == 'im':
        question = message['text']
        response = query_sambaqa(question)
        say(f"SambaQA response:\n{response}")

# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, settings.SLACK_APP_TOKEN).start()