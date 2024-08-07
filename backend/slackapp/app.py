import os
import sys
import logging
import json
from dotenv import load_dotenv
import re
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup path and environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))
sys.path.extend([kit_dir, repo_dir])

env_path = os.path.join(repo_dir, '.env')
load_dotenv(env_path, override=True)

# Initialize Slack app
app = App(token=os.getenv("SLACK_TOKEN", ""))

# SambaQA API configuration
SAMBAQA_API_URL = os.getenv("SAMBAQA_API_URL", "http://localhost:8000/retrievers/basic-rag/answer")
DEFAULT_COLLECTION = "snspubdocumentation"
DEFAULT_MODEL = "sambastudio/Meta-Llama-3-70B-Instruct"

def clean_input(text):
    # Remove Slack-style user mentions
    text = re.sub(r'^<@U[A-Z0-9]+>:?\s*', '', text.strip())
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    
    # Remove or replace problematic Unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove any non-ASCII characters
    text = ''.join(char for char in text if ord(char) < 128)
    
    # Replace potentially problematic characters
    original_text = text
    problematic_chars = {'{': '(', '}': ')', '[': '(', ']': ')', '"': "'", "'": "'"}
    for char, replacement in problematic_chars.items():
        text = text.replace(char, replacement)
    
    if text != original_text:
        logger.info(f"Input contained problematic characters. Original: '{original_text}', Cleaned: '{text}'")
        return text, True
    
    return text, False

def validate_input(text):
    if len(text) < 1 or len(text) > 1000:  # Adjust these limits as needed
        raise ValueError("Input must be between 1 and 1000 characters")
    
    if not text.strip():
        raise ValueError("Input cannot be empty")
    
    return text

def query_sambaqa(question):
    logger.info(f"Original question: {question}")
    
    try:
        cleaned_question, was_modified = clean_input(question)
        logger.info(f"Cleaned question: {cleaned_question}")
        
        validated_question = validate_input(cleaned_question)
        logger.info(f"Validated question: {validated_question}")

        modification_message = "" if was_modified else ""

        payload = {
            "collection_name": DEFAULT_COLLECTION,
            "query": validated_question,
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

        response = requests.post(SAMBAQA_API_URL, json=payload)
        response.raise_for_status()
        
        if not response.text.strip():
            logger.error("Received empty response from SambaQA API")
            return f"{modification_message}I apologize, but I received an empty response from my knowledge base. Could you please rephrase your question or try again later?"
        
        try:
            json_response = response.json()
        except json.JSONDecodeError as json_err:
            logger.error(f"JSON Decode Error: {str(json_err)}")
            logger.error(f"Response content: {response.text}")
            return f"{modification_message}I'm sorry, but I'm having trouble understanding the response from my knowledge base. Could you please try asking your question again in a different way?"
        
        if 'answer' not in json_response:
            logger.error(f"Unexpected response structure: {json_response}")
            return f"{modification_message}I apologize, but I received an unexpected response format. Could you please rephrase your question or try again later?"
        
        return f"{modification_message}{json_response['answer']}"
    except ValueError as e:
        logger.error(f"Input validation error: {str(e)}")
        return f"I'm sorry, but there seems to be an issue with your input: {str(e)}. Could you please try rephrasing your question?"
    except requests.RequestException as e:
        logger.error(f"Error querying SambaQA: {str(e)}")
        return "I apologize, but I'm having trouble connecting to my knowledge base at the moment. Could you please try again in a few minutes?"
    except Exception as e:
        logger.error(f"Unexpected error in query_sambaqa: {str(e)}", exc_info=True)
        return "I'm sorry, but an unexpected error occurred while processing your request. Could you please try again later or rephrase your question?"


@app.event("app_mention")
def handle_app_mention_events(body, say):
    try:
        question = body["event"]["text"]
        response = query_sambaqa(question)
        say(response)
        logger.info(f"App mention event body: {body}")
    except Exception as e:
        error_message = "I apologize, but I'm experiencing some difficulties at the moment. Our team is working on resolving the issue. Could you please try again later?"
        logger.error(f"Error in handle_app_mention_events: {str(e)}", exc_info=True)
        say(error_message)

if __name__ == "__main__":
    try:
        logger.info("Starting Slack app...")
        SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN", "")).start()
    except Exception as e:
        logger.error(f"Error starting Slack app: {str(e)}", exc_info=True)